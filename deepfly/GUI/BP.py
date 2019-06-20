import itertools

import numpy as np

from .Config import config
from .Camera import Camera
from .util.optim_util import project_on_last, energy_drosoph
from .Config import config

class LegBP:
    def __init__(
        self,
        camera_network,
        img_id,
        j_id_list,
        bone_param=config["bone_param"],
        num_peak=config["num_peak"],
        prior=None,
        upper_bound=config["upper_bound"],
        image_shape=config["image_shape"],
    ):
        self.camera_network = camera_network
        self.cam_list = self.camera_network.cam_list
        self.img_id = img_id
        self.j_id_list = j_id_list
        self.bone_param = bone_param
        self.num_peak = num_peak
        self.upper_bound = upper_bound
        self.cam_id_list = [cam.cam_id for cam in self.camera_network]

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
            # find 2d proposals for a given joint for each camera, by taking local maximums
            for cam in self.camera_network:
                min_distance = 1
                threshold_abs = 0.0
                '''
                threshold_rel = (
                    0.95
                    if config["skeleton"].is_tracked_point(j.j_id, )
                       or config["skeleton"].is_coxa_femur(j.j_id)
                       or config["skeleton"].is_antenna(j.j_id)
                    else 0.1
                )
                '''
                threshold_rel = 0.5

                p2d_list.append(
                    Camera.hm_to_pred(
                        np.squeeze(cam.get_heatmap(self.img_id, j.j_id)),
                        num_pred=num_peak,
                        min_distance=min_distance,
                        threshold_abs=threshold_abs,
                        threshold_rel=threshold_rel,
                    )
                )

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
                for cam, p2d in zip(self.camera_network.cam_list, p2d_prop):
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
                        self.camera_network.cam_list,
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
