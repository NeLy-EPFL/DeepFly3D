from __future__ import absolute_import, print_function

import glob
import json
import os
import pickle
import re
from os.path import isfile

import numpy as np
import scipy
import torch
import torch.utils.data as data
from skimage import transform

import deepfly.logger as logger
from deepfly.Config import config

# from deepfly.os_util import *
from deepfly.os_util import constr_img_name, parse_img_name, read_camera_order
from deepfly.pose2d.utils.transforms import (
    color_normalize,
    draw_labelmap,
    fliplr,
    im_to_torch,
    load_image,
    random_jitter,
    random_rotation,
    resize,
    shufflelr,
    to_torch,
    im_to_numpy
)

FOLDER_NAME = 0
IMAGE_NAME = 1


def read_json(d, json_path, folder_train_list, cidread2cid):
    json_data = json.load(open(json_path, "r"))
    for session_id in json_data.keys():
        for folder_name in json_data[session_id]["data"].keys():
            if folder_name not in folder_train_list:
                continue
            for image_name in json_data[session_id]["data"][folder_name].keys():
                key = ("/data/annot/" + folder_name, image_name)
                # for the hand annotations, it is always correct ordering
                cidread2cid[key[FOLDER_NAME]] = np.arange(config["num_cameras"])
                cid_read, _ = parse_img_name(image_name)
                if cid_read == 3:
                    continue
                pts = json_data[session_id]["data"][folder_name][image_name]["position"]
                d[key] = np.array(pts)


def find_pose_corr_recursively(path):
    return list(glob.glob(os.path.join(path, "./**/pose_corr*.pkl"), recursive=True))


def read_manual_corrections(
    d, output_folder, manual_path_list, cidread2cid_global, num_classes
):
    pose_corr_path_list = []
    for root in manual_path_list:
        logger.debug("Searching recursively: {}".format(root))
        pose_corr_path_list.extend(find_pose_corr_recursively(root))
    logger.debug(
        "Number of manual correction files: {}".format(len(pose_corr_path_list))
    )
    for path in pose_corr_path_list:
        d = pickle.load(open(path, "rb"))
        folder_name = d["folder"]
        key_folder_name = folder_name
        if folder_name not in cidread2cid_global:
            cam_folder = os.path.join(folder_name, output_folder)
            cidread2cid, cid2cidread = read_camera_order(cam_folder)
            cidread2cid_global[key_folder_name] = cidread2cid
        for cid in range(config["num_cameras"]):
            for img_id, points2d in d[cid].items():
                cid_read = cidread2cid[key_folder_name].tolist().index(cid)
                key = (key_folder_name, constr_img_name(cid_read, img_id))
                num_heatmaps = points2d.shape[0]

                pts = np.zeros((2 * num_classes, 2), dtype=np.float)
                if cid < 3:
                    pts[: num_heatmaps // 2, :] = points2d[: num_heatmaps // 2, :]
                elif 3 < cid < 7:
                    pts[num_classes : num_classes + (num_heatmaps // 2), :] = points2d[
                        num_heatmaps // 2 :, :
                    ]
                elif cid == 3:
                    continue
                else:
                    raise NotImplementedError

                d[key] = pts


def read_unlabeled_folder(d, unlabeled, output_folder, cidread2cid_global, max_img_id):
    calib_folder = os.path.join(unlabeled, output_folder)
    cidread2cid, _ = read_camera_order(calib_folder)
    cidread2cid_global[unlabeled] = cidread2cid

    for image_name_jpg in os.listdir(unlabeled):
        match = re.match("camera_(\d+)_img_(\d+)\.jpg", image_name_jpg)
        if match:
            image_name = image_name_jpg.replace(".jpg", "")
            key = (unlabeled, image_name)
            cid_read, img_id = parse_img_name(image_name)
            if cidread2cid.tolist().index(cid_read) == 3:
                continue
            if max_img_id is not None and img_id > max_img_id:
                continue
            d[key] = np.zeros(shape=(config["skeleton"].num_joints, 2))


def normalize_annotations(d, num_classes, cidread2cid_global):
    """
    There are three cases:
    30: 15x2 5 points in each 3 legs, on 2 sides
    32: 15 tracked points,, plus antenna on each side
    38: 15 tracked points, then 3 stripes, then one antenna
    """
    for k, v in d.copy().items():
        cid_read, img_id = parse_img_name(k[IMAGE_NAME])
        folder_name = k[FOLDER_NAME]
        cid = cidread2cid_global[folder_name][cid_read] if cid_read != 7 else 3
        if "annot" in k[FOLDER_NAME]:
            if cid > 3:  # then right camera
                v[:15, :] = v[15:30:]
                v[15] = v[35]  # 35 is the second antenna
            elif cid < 3:
                v[15] = v[30]  # 30 is the first antenna
            else:
                raise NotImplementedError
            v[16:, :] = 0.0
        else:  # then manual correction
            if cid > 3:  # then right camera
                v[:num_classes, :] = v[num_classes:, :]
            if cid == 3:
                continue
            v = v[:num_classes, :]  # keep only one side

        j_keep = np.arange(0, num_classes)
        v = v[j_keep, :]
        v = np.abs(v)  # removing low-confidence
        assert np.logical_or(0 <= v, v <= 1).all()

        d[k] = v


class DrosophilaDataset(data.Dataset):
    def __init__(
        self,
        data_folder,
        data_corr_folder=None,
        img_res=None,
        hm_res=None,
        train=True,
        sigma=1,
        jsonfile="drosophilaimaging-export.json",
        session_id_train_list=None,
        folder_train_list=None,
        augmentation=False,
        evaluation=False,
        unlabeled=None,
        num_classes=config["num_predict"],
        max_img_id=None,
        output_folder=None,
    ):
        self.train = train
        self.data_folder = data_folder  # root image folders
        self.data_corr_folder = data_corr_folder
        self.json_file = os.path.join(jsonfile)
        self.is_train = train  # training set or test set
        self.img_res = img_res
        self.hm_res = hm_res
        self.sigma = sigma
        self.augmentation = augmentation
        self.evaluation = evaluation
        self.unlabeled = unlabeled
        self.num_classes = num_classes
        self.max_img_id = max_img_id
        self.cidread2cid = dict()
        self.output_folder = output_folder
        self.manual_path_list = []
        self.session_id_train_list = session_id_train_list
        self.folder_train_list = folder_train_list
        if self.output_folder is None:
            raise ValueError(
                "Please provide an output_folder relative to images folder"
            )

        assert (
            not self.evaluation or not self.augmentation
        )  # self eval then not augmentation
        assert not self.unlabeled or evaluation  # if unlabeled then evaluation

        self.annotation_dict = dict()

        if isfile(self.json_file) and not self.unlabeled:
            logger.debug("Searching for json file")
            read_json(
                self.annotation_dict,
                self.json_file,
                self.folder_train_list,
                self.cidread2cid,
            )

        if not self.unlabeled and self.train and self.manual_path_list:
            logger.debug("Searching for manual corrections")
            read_manual_corrections(
                self.annotation_dict,
                self.output_folder,
                self.manual_path_list,
                self.cidread2cid,
                self.num_classes,
            )

        if self.unlabeled:
            logger.debug("Searching unlabeled")
            read_unlabeled_folder(
                self.annotation_dict,
                self.unlabeled,
                self.output_folder,
                self.cidread2cid,
                self.max_img_id,
            )

        # make sure data is in the folder
        for folder_name, image_name in self.annotation_dict.copy().keys():
            image_file = os.path.join(
                self.data_folder,
                folder_name.replace("_network", ""),
                image_name + ".jpg",
            )

            if not os.path.isfile(image_file):
                self.annotation_dict.pop((folder_name, image_name), None)
                print("FileNotFound: {}/{} ".format(folder_name, image_name))

        normalize_annotations(self.annotation_dict, self.num_classes, self.cidread2cid)

        self.annotation_key = list(self.annotation_dict.keys())
        if self.evaluation:  # sort keys
            self.annotation_key.sort(
                key=lambda x: x[0] + "_" + x[1].split("_")[3] + "_" + x[1].split("_")[1]
            )

        self.mean, self.std = self._compute_mean()

        logger.debug(
            "Folders inside {} data: {}".format(
                "train" if self.train else "validation",
                set([k[0] for k in self.annotation_key]),
            )
        )
        logger.debug(
            "Successfully imported {} Images in Drosophila Dataset".format(len(self))
        )

    def _compute_mean(self):
        meanstd_file = config["mean"]
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            raise FileNotFoundError
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for k in self.annotation_key:
                img_path = os.path.join(
                    self.data_folder, k[FOLDER_NAME], k[IMAGE_NAME] + ".jpg"
                )
                img = load_image(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self)
            std /= len(self)
            meanstd = {"mean": mean, "std": std}
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            logger.debug(
                "    Mean: %.4f, %.4f, %.4f"
                % (meanstd["mean"][0], meanstd["mean"][1], meanstd["mean"][2])
            )
            logger.debug(
                "    Std:  %.4f, %.4f, %.4f"
                % (meanstd["std"][0], meanstd["std"][1], meanstd["std"][2])
            )

        return meanstd["mean"], meanstd["std"]

    def __get_image_path(self, folder_name, camera_id, pose_id, pad=True):
        img_path = os.path.join(
            self.data_folder,
            folder_name.replace("_network", ""),
            constr_img_name(camera_id, pose_id, pad=pad) + ".jpg",
        )
        return img_path

    def __getitem__(self, index, batch_mode=True, temporal=False):
        folder_name, img_name = (
            self.annotation_key[index][FOLDER_NAME],
            self.annotation_key[index][IMAGE_NAME],
        )
        cid_read, pose_id = parse_img_name(img_name)
        cid = self.cidread2cid[folder_name][cid_read]
        flip = cid in config["flip_cameras"] and (
            "annot" in folder_name or ("annot" not in folder_name and self.unlabeled)
        )

        try:
            img_orig = load_image(self.__get_image_path(folder_name, cid_read, pose_id))
        except FileNotFoundError:
            try:
                img_orig = load_image(
                    self.__get_image_path(folder_name, cid_read, pose_id, pad=False)
                )
            except FileNotFoundError:
                print(
                    "Cannot read index {} {} {} {}".format(
                        index, folder_name, cid_read, pose_id
                    )
                )
                return self.__getitem__(index + 1)

        pts = torch.Tensor(self.annotation_dict[self.annotation_key[index]])
        nparts = pts.size(0)
        assert nparts == config["num_predict"]

        joint_exists = np.zeros(shape=(nparts,), dtype=np.uint8)
        for i in range(nparts):
            # we convert to int as we cannot pass boolean from pytorch dataloader
            # as we decrease the number of joints to skeleton.num_joints during training
            joint_exists[i] = (
                1
                if (
                    (0.01 < pts[i][0] < 0.99)
                    and (0.01 < pts[i][1] < 0.99)
                    and (
                        config["skeleton"].camera_see_joint(cid, i)
                        or config["skeleton"].camera_see_joint(
                            cid, (i + config["num_predict"])
                        )
                    )
                )
                else 0
            )
        if flip:
            img_orig = torch.from_numpy(fliplr(img_orig.numpy())).float()
            pts = shufflelr(pts, width=img_orig.size(2), dataset="drosophila")
        if img_orig.shape[0] == 3:
            img_orig = im_to_numpy(img_orig)
        img_norm = im_to_torch(transform.resize(img_orig, self.img_res))  # scipy.misc.imresize

        # Generate ground truth heatmap
        tpts = pts.clone()
        target = torch.zeros(nparts, self.hm_res[0], self.hm_res[1])

        for i in range(nparts):
            if joint_exists[i] == 1:
                tpts = to_torch(
                    pts * to_torch(np.array([self.hm_res[1], self.hm_res[0]])).float()
                )
                target[i] = draw_labelmap(
                    target[i], tpts[i], self.sigma, type="Gaussian"
                )
            else:
                # to make invisible joints explicit in the training visualization
                # these values are not used to calculate loss
                target[i] = torch.ones_like(target[i])

        # augmentation
        if self.augmentation:
            img_norm = random_jitter(
                img_norm, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
            )
            img_norm, target = random_rotation(img_norm, target, degrees=10)

        img_norm = color_normalize(img_norm, self.mean, self.std)

        if cid == 3 or cid == 7:
            raise NotImplementedError
        meta = {
            "inp": resize(img_orig, 600, 350),
            "folder_name": folder_name,
            "image_name": img_name,
            "index": index,
            "center": 0,
            "scale": 0,
            "pts": pts,
            "tpts": tpts,
            "cid": cid,
            "cam_read_id": cid_read,
            "pid": pose_id,
            "joint_exists": joint_exists,
        }

        return img_norm, target, meta

    def greatest_image_id(self):
        ids = [parse_img_name(k[1])[1] for k in self.annotation_key]
        return max(ids) if ids else 0

    def __len__(self):
        return len(self.annotation_key)
