from __future__ import absolute_import, print_function

import glob
import json
import os
import pickle
from os.path import isfile

import numpy as np
import scipy
import torch
import torch.utils.data as data

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
)

FOLDER_NAME = 0
IMAGE_NAME = 1


def find_pose_corr_recursively(path):
    return list(glob.glob(os.path.join(path, "./**/pose_corr*.pkl"), recursive=True))


def read_unlabeled_folder(
    d, unlabeled, output_folder, cidread2cid_global, max_img_id, front
):
    calib_folder = os.path.join(unlabeled, output_folder)
    cidread2cid, _ = read_camera_order(calib_folder)
    cidread2cid_global[unlabeled] = cidread2cid

    for image_name_jpg in os.listdir(unlabeled):
        if image_name_jpg.endswith(".jpg"):
            key = unlabeled + "/" + image_name_jpg
            cid_read, img_id = parse_img_name(image_name_jpg.replace(".jpg", ""))
            if not front:
                if cidread2cid.tolist().index(cid_read) == 3:
                    continue
            else:
                if cidread2cid.tolist().index(cid_read) != 3:
                    continue
            if max_img_id is not None and img_id > max_img_id:
                continue
            d[key] = np.zeros(shape=(config["num_predict"], 2))


class DrosophilaDataset(data.Dataset):
    def __init__(
        self,
        data_folder,
        img_res=None,
        hm_res=None,
        train=True,
        augmentation=False,
        unlabeled=None,
        num_classes=config["num_predict"],
        max_img_id=None,
        output_folder=None,
        front=True,
    ):
        self.train = train
        self.data_folder = data_folder  # root image folders
        self.is_train = train  # training set or test set
        self.img_res = img_res
        self.hm_res = hm_res
        self.sigma = 1
        self.augmentation = augmentation
        self.unlabeled = unlabeled
        self.num_classes = num_classes
        self.max_img_id = max_img_id
        self.cidread2cid = dict()
        self.output_folder = output_folder
        self.manual_path_list = []
        self.front = front
        if self.output_folder is None:
            raise ValueError(
                "Please provide an output_folder relative to images folder"
            )
        self.annotation_dict = dict()

        if self.unlabeled:
            logger.debug(f"Searching for images under folder {unlabeled}")
            read_unlabeled_folder(
                self.annotation_dict,
                self.unlabeled,
                self.output_folder,
                self.cidread2cid,
                self.max_img_id,
                self.front,
            )
        else:
            self.annotation_dict = pickle.load(open(self.data_folder, "rb"))
            for k in self.annotation_dict.keys():
                self.cidread2cid[os.path.split(k)[0]] = np.arange(7)

        # make sure data is in the folder
        for image_path in self.annotation_dict.copy().keys():
            if not os.path.isfile(image_path):
                self.annotation_dict.pop(image_path, None)
                print("FileNotFound: {} ".format(image_path))

        self.annotation_key = list(self.annotation_dict.keys())

        self.mean, self.std = self._compute_mean()

        logger.debug(
            "Successfully imported {} Images in Drosophila Dataset".format(len(self))
        )

    def _compute_mean(self):
        meanstd_file = config["mean"]
        meanstd = torch.load(meanstd_file)

        return meanstd["mean"], meanstd["std"]

    def __get_image_path(self, folder_name, camera_id, pose_id, pad=True):
        img_path = os.path.join(
            folder_name.replace("_network", ""),
            constr_img_name(camera_id, pose_id, pad=pad) + ".jpg",
        )
        return img_path

    def __getitem__(self, index):
        folder_name, img_name = os.path.split(self.annotation_key[index])

        cid_read, pose_id = parse_img_name(img_name.replace(".jpg", ""))
        cid = self.cidread2cid[folder_name][cid_read]
        flip = cid in config["flip_cameras"] and (
            "annot" in folder_name or ("annot" not in folder_name and self.unlabeled)
        )

        try:
            nm = self.__get_image_path(folder_name, cid_read, pose_id, pad=False)
            nm_pad = self.__get_image_path(folder_name, cid_read, pose_id, pad=True)
            img_orig = load_image(nm)
        except FileNotFoundError:
            try:
                img_orig = load_image(nm_pad)
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

        joint_exists = np.ones(shape=(nparts,))
        if not self.front:
            for i in range(nparts):
                can_see = config["skeleton"].camera_see_joint(cid, i) or config[
                    "skeleton"
                ].camera_see_joint(cid, (i + config["num_predict"]))
                can_see = int(can_see)
                joint_exists[i] = int(torch.all(pts[i] != 0)) * can_see
        else:
            for i in range(nparts):
                joint_exists[i] = int(torch.all(pts[i] != 0))

        if flip:
            img_orig = torch.from_numpy(fliplr(img_orig.numpy())).float()
            pts = shufflelr(pts, width=img_orig.size(2), dataset="drosophila")
        img_norm = im_to_torch(scipy.misc.imresize(img_orig, self.img_res))

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

        # augmentation
        if self.augmentation:
            img_norm = random_jitter(
                img_norm, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
            )
            img_norm, target = random_rotation(img_norm, target, degrees=15)

        img_norm = color_normalize(img_norm, self.mean, self.std)

        meta = {
            "inp": resize(img_orig, 600, 350),
            "folder_name": folder_name,
            "image_name": img_name,
            "index": index,
            "pts": pts,
            "tpts": tpts,
            "cid": cid,
            "cam_read_id": cid_read,
            "pid": pose_id,
            "joint_exists": joint_exists,
        }

        return img_norm, target, meta

    def greatest_image_id(self):
        # print(os.path.splitext(self.annotation_key[0]))
        ids = [parse_img_name(os.path.split(k)[1])[1] for k in self.annotation_key]
        return max(ids) if ids else 0

    def __len__(self):
        return len(self.annotation_key)

