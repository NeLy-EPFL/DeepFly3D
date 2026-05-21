"""
Pictorial-structures pose correction (Figure 10 of the DeepFly3D eLife
paper). Resurrected from the pre-2021 `deepfly.belief_propagation` module.

`solve_belief_propagation` expects each `pyba.Camera` in `cam_list` to
carry both `cam.cam_id` (used by skeleton visibility checks) and
`cam.heatmaps` (per-joint probability maps as a (n_frames, n_joints,
H, W) array). `df3d.core.Core.run_belief_propagation` attaches the
network-produced heatmaps to each camera before invoking this module.

Coordinate convention: 2D points throughout this module are normalized
to [0, 1] in (x, y) order -- the convention used by `pyba.Camera` and
`df3d.optim_util.energy_drosoph`. `df2d.util.heatmap_peaks` natively
returns (y, x); `_top_k_peaks_xy_normalized` below swaps the last axis
at the boundary.
"""
import itertools

import numpy as np

from df2d.util import heatmap_peaks

from df3d import logger
from df3d.config import config
from df3d.optim_util import project_on_last, energy_drosoph


def _top_k_peaks_xy_normalized(heatmap_2d, num_peak, min_distance=1,
                               threshold_rel=0.5):
    """Extract up to `num_peak` local-maximum peaks from a single heatmap.

    Returns a list of (x_norm, y_norm) arrays, dropping any padded entries.
    `df2d.util.heatmap_peaks` returns (y, x, score) per peak; this helper
    wraps it for the (x, y) convention used inside this module.
    """
    if heatmap_2d is None:
        return []
    peaks = heatmap_peaks(
        heatmap_2d[np.newaxis, np.newaxis], k=num_peak,
        min_distance=min_distance, threshold_rel=threshold_rel,
    )[0, 0]  # (K, 3) = [(y_norm, x_norm, score)]
    real = peaks[peaks[:, 2] > 0]
    if real.size == 0:
        return []
    return [np.array([row[1], row[0]], dtype=float) for row in real]


def solve_belief_propagation(cam_list, img_id, bone_param, num_peak=10, prior=None):
        # Every camera must carry an integer cam_id: it drives the skeleton
        # visibility lookups and candidate indexing below.
        if any(not isinstance(c.cam_id, (int, np.integer)) for c in cam_list):
            raise ValueError(
                'solve_belief_propagation requires every camera to have an '
                'integer cam_id, but got cam_id values '
                f'{[c.cam_id for c in cam_list]}.'
            )

        # find all the connected parts
        j_id_list_list = [
            [j for j in range(config["skeleton"].num_joints) if config["skeleton"].limb_id[j] == limb_id]
            for limb_id in range(config["skeleton"].num_limbs)
        ]

        chain_list = list()
        for j_id_l in j_id_list_list:
            visible = np.zeros(shape=(len(j_id_l),), dtype=int)
            for cam in cam_list:
                visible += [
                    config["skeleton"].camera_see_joint(cam.cam_id, j_id) for j_id in j_id_l
                ]
            if np.all(visible >= 2):
                chain_list.append(
                    LegBP(
                        cam_list=cam_list,
                        img_id=img_id,
                        j_id_list=j_id_l,
                        bone_param=bone_param,
                        num_peak=num_peak,
                        prior=prior,
                    )
                )
            else:
                pass
                # logger.debug("Joints {} is not visible from at least two cameras".format(j_id_l))

        logger.debug([
                [len(leg[i].candid_list) for i in range(len(leg.jointbp))]
                for leg in chain_list
            ])

        for chain in chain_list:
            chain.propagate()
            chain.solve()

        # read the best 2d locations
        points2d_list = [
            np.zeros((config["skeleton"].num_joints, 2), dtype=float)
            for _ in range(len(cam_list))
        ]
        for leg in chain_list:
            for cam_idx in range(len(cam_list)):
                for idx, j_id in enumerate(leg.j_id_list):
                    points2d_list[cam_idx][j_id] = leg[idx][leg[idx].argmin].p2d[
                        cam_idx
                    ]

        return points2d_list.copy()


class LegBP:
    def __init__(
        self,
        cam_list,
        img_id,
        j_id_list,
        bone_param=None,
        num_peak=None,
        prior=None,
        upper_bound=None,
        image_shape=None,
    ):
        # Resolve defaults lazily; `config["image_shape"]` is only populated
        # after `df3d.core.Core` reads an image at runtime.
        if bone_param is None:
            bone_param = config['bone_param']
        if num_peak is None:
            num_peak = config['num_peak']
        if upper_bound is None:
            upper_bound = config['upper_bound']
        if image_shape is None:
            image_shape = config['image_shape']

        self.cam_list = cam_list
        self.img_id = img_id
        self.j_id_list = j_id_list
        self.bone_param = bone_param
        self.num_peak = num_peak
        self.upper_bound = upper_bound
        self.cam_id_list = [cam.cam_id for cam in self.cam_list]

        self.image_res = image_shape
        self.jointbp = [JointBP(j_id) for j_id in j_id_list]
        self.prior = prior
        self.generate_proposals(self.num_peak, prior)

        self.alpha_reproj = config["alpha_reproj"]
        self.alpha_heatmap = config["alpha_heatmap"]
        self.alpha_bone = config["alpha_bone"]

    def __getitem__(self, i):
        return self.jointbp[i]

    def generate_proposals(self, num_peak, prior=None):
        for j in self.jointbp:
            cam_id_list_seeing_joint = [
                cam_id
                for cam_id in self.cam_id_list
                if config["skeleton"].camera_see_joint(cam_id, j.j_id)
            ]
            p2d_list = []
            # find 2d proposals for a given joint for each camera, by taking
            # local maximums. For cameras that don't see this joint (or whose
            # heatmap is all-zero), append a single placeholder peak so that
            # itertools.product(*p2d_list) stays non-empty; the placeholder is
            # filtered out downstream via config["skeleton"].camera_see_joint.
            placeholder = [np.array([0.0, 0.0])]
            for cam in self.cam_list:
                if not config["skeleton"].camera_see_joint(cam.cam_id, j.j_id):
                    p2d_list.append(placeholder)
                    continue
                if cam.heatmaps is None:
                    p2d_list.append(placeholder)
                    continue
                hm = cam.heatmaps[self.img_id, j.j_id]
                peaks = _top_k_peaks_xy_normalized(
                    hm, num_peak=num_peak,
                    min_distance=1, threshold_rel=0.5,
                )
                p2d_list.append(peaks if len(peaks) > 0 else placeholder)

            # set the priors (user manual correction)
            cams_with_prior = []
            if prior is not None:
                for cam_id, joint_id, pts in prior:
                    if joint_id == j.j_id:
                        cam_index = self.cam_id_list.index(cam_id)
                        p2d_list[cam_index] = [
                            pts
                        ]  # we remove all the other heatmap proposals
                        cams_with_prior.append(cam_id)

            # find 3d proposals by triangulating with all the visible cameras
            for p2d_prop in list(itertools.product(*p2d_list)):
                if j.get_num_candid() > self.upper_bound:
                    # print("Hit upper bound of 3d proposals {}".format(self.upper_bound))
                    continue

                p2d_list_iter = list()
                cam_list_iter = list()
                for cam, p2d in zip(self.cam_list, p2d_prop):
                    if config["skeleton"].camera_see_joint(cam.cam_id, j.j_id):
                        p2d_list_iter.append(p2d)
                        cam_list_iter.append(cam)
                p2d_list_iter = np.array(p2d_list_iter).reshape(-1, 2)
                p3d, err_proj, prob_hm, _ = energy_drosoph(
                    cam_list_iter, self.img_id, j.j_id, p2d_list_iter, None, None
                )
                prob_hm += len(
                    [
                        cam_id
                        for cam_id in cams_with_prior
                        if config["skeleton"].camera_see_joint(cam_id, j.j_id)
                    ]
                )
                # we give p2d_prop instead of p2d, as we need to set 2d values also for invisible points.
                j.add_candid(p3d, p2d_prop, err_proj, prob_hm)

            # for every triplet of cameras seeing the point
            for camid_x, camid_y, camid_z in itertools.permutations(
                cam_id_list_seeing_joint, 3
            ):
                cam_index_x, cam_index_y, cam_index_z = (
                    self.cam_id_list.index(camid_x),
                    self.cam_id_list.index(camid_y),
                    self.cam_id_list.index(camid_z),
                )
                cam_x, cam_y, cam_z = (
                    self.cam_list[cam_index_x],
                    self.cam_list[cam_index_y],
                    self.cam_list[cam_index_z],
                )
                # iterate over all combinations of 2d proposals
                for p2d_x, p2d_y, p2d_z in list(
                    itertools.product(
                        *[
                            p2d_list[cam_index_x],
                            p2d_list[cam_index_y],
                            p2d_list[cam_index_z],
                        ]
                    )
                ):
                    p2d_list_iter = [None] * len(self.cam_id_list)
                    p2d_list_iter[cam_index_x] = p2d_x
                    p2d_list_iter[cam_index_y] = p2d_y
                    p2d_list_iter[cam_index_z] = p2d_z

                    # for all remaining cameras triangulate and project to find 2d points
                    for cam_index_project in [
                        cam_index
                        for cam_index in range(len(self.cam_id_list))
                        if cam_index not in [cam_index_x, cam_index_y, cam_index_z]
                    ]:
                        p2d_list_iter[cam_index_project] = (
                            project_on_last(
                                [cam_x, cam_y, cam_z, self.cam_list[cam_index_project]],
                                np.array([p2d_x, p2d_y, p2d_z]).reshape(-1, 2)
                                * self.image_res,
                            )
                            / self.image_res
                        )
                    p2d_list_iter = np.array(p2d_list_iter).reshape(-1, 2)
                    p3d, err_proj, prob_hm, _ = energy_drosoph(
                        self.cam_list,
                        self.img_id,
                        j.j_id,
                        p2d_list_iter,
                        None,
                        None,
                    )
                    prob_hm += len(
                        [
                            cam_id
                            for cam_id in cams_with_prior
                            if (
                                config["skeleton"].camera_see_joint(cam_id, j.j_id)
                                and cam_id in [camid_x, camid_y, camid_z]
                            )
                        ]
                    )

                    j.add_candid(p3d, p2d_list_iter, err_proj, prob_hm)

    def propagate(self):  # start from the leaf, calculate belief for each candid
        for c in self.jointbp[-1].candid_list:  # the only leaf
            c.belief = 1
        for idx in range(
            len(self.j_id_list) - 1, 0, -1
        ):  # finish at the root (at 0) (excluding)
            for cand_p in self.jointbp[idx - 1].candid_list:
                j_c = self.jointbp[idx]
                cand_p.belief = min(
                    [self.B_j(cand_c, cand_p) for cand_c in j_c.candid_list]
                )

    def B_j(self, cand_c, cand_p):  # function of parent node, cost function
        return (
            self.m_j(cand_c)
            + self.alpha_bone
            * (
                1
                - self.d_ij(
                    cand_c.p3d, cand_p.p3d, self.bone_param[cand_p.j_id], cand_p.j_id
                )
            )
            + cand_c.belief
        )

    def d_ij(self, p3d_c, p3d_p, param, joint_id):
        mu, sig = param
        if np.isnan(mu) or np.isnan(sig):
            raise Exception
        dist = np.linalg.norm((p3d_c - p3d_p))
        return np.exp(-np.power(dist - mu, 2.0) / (2 * np.power(sig, 2.0)))

        # if joint tarsus tip, then penalize less for shrinking
        '''
        if joint_id % 5 != 4:
            return np.exp(-np.power(dist - mu, 2.0) / (2 * np.power(sig, 2.0)))
        else:
            if dist > mu + sig:
                return np.exp(-np.power(dist - mu, 2.0) / (2 * np.power(sig / 5, 2.0)))
            else:
                return np.exp(-np.power(dist - mu, 2.0) / (2 * np.power(sig * 3, 2.0)))
        '''

    def m_j(self, cand):
        return self.alpha_reproj * cand.err_proj + self.alpha_heatmap * (
            1 - cand.prob_hm
        )

    def solve(self):  # start from the root, set the joint with the largest belief
        root_j = self.jointbp[0]
        belief_list = [
            self.m_j(candid) + candid.belief for candid in root_j.candid_list
        ]
        root_j.argmin = np.argmin(belief_list)

        cand_p = root_j.candid_list[root_j.argmin]
        for idx in range(1, len(self.j_id_list)):  # exclude root, already solved
            self.jointbp[idx].argmin = np.argmin(
                [self.B_j(cand_c, cand_p) for cand_c in self.jointbp[idx].candid_list]
            )
            cand_p = self.jointbp[idx].candid_list[self.jointbp[idx].argmin]


class JointBP:
    def __init__(self, j_id):
        self.j_id = j_id
        self.candid_list = list()
        self.argmin = None

    def __getitem__(self, i):
        return self.candid_list[i]

    def get_num_candid(self):
        return len(self.candid_list)

    def add_candid(self, p3d, p2d, err_proj, prob_hm, belief=1):
        self.candid_list.append(Candid(self.j_id, p3d, p2d, err_proj, prob_hm))


class Candid:
    def __init__(self, j_id, p3d, p2d, err_proj, prob_hm, belief=1):
        # print(err_proj, prob_hm)
        self.j_id = j_id
        self.p3d = p3d
        self.p2d = p2d
        self.err_proj = err_proj
        self.prob_hm = prob_hm
        self.belief = belief
