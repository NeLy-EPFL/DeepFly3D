import glob
import os
import pickle

import numpy as np
import copy

from df3d.config import config


class PoseDB:
    def __init__(self, folder, meta=None):
        self.folder = folder

        self.db_path_list = glob.glob(os.path.join(self.folder, "pose_corr*.pkl"))
        self.last_write_image_id = 0
        if len(self.db_path_list) != 0:
            self.db_path = self.db_path_list[0]
            self.db = pickle.load(open(self.db_path, "rb"))
        else:
            self.db_path = os.path.join(
                self.folder, "pose_corr_{}.pkl".format(self.folder.replace("/", "-"))
            )
            self.db = {i: dict() for i in range(config["num_cameras"])}
            self.db["folder"] = self.folder
            self.db["meta"] = meta
            self.db["train"] = {i: dict() for i in range(config["num_cameras"])}
            self.db["modified"] = {i: dict() for i in range(config["num_cameras"])}

            self.dump()

    def read(self, cam_id, img_id):
        if img_id in self.db[cam_id]:
            return np.array(self.db[cam_id][img_id])
        else:
            return None

    def read_modified_joints(self, cam_id, img_id):
        if img_id in self.db["modified"][cam_id]:
            return self.db["modified"][cam_id][img_id]
        else:
            return []

    def write(self, pts, cam_id, img_id, train, modified_joints):
        assert pts.shape[0] == config["skeleton"].num_joints and pts.shape[1] == 2
        assert modified_joints is not None

        self.db[cam_id][img_id] = pts

        self.db["train"][cam_id][img_id] = train
        self.db["modified"][cam_id][img_id] = modified_joints

        self.last_write_image_id = img_id

    def remove_corrections(self, cam_id, img_id):
        if img_id in self.db.get(cam_id, {}):
            del self.db[cam_id][img_id]
        #
        if img_id in self.db['train'].get(cam_id, {}):
            del self.db["train"][cam_id][img_id]
        #
        if img_id in self.db['modified'].get(cam_id, {}):
            del self.db["modified"][cam_id][img_id]

    def dump(self):
        with open(self.db_path, "wb") as outfile:
            pickle.dump(self.db, outfile)

    def has_key(self, cam_id, img_id):
        return img_id in self.db[cam_id]

    def manual_corrections(self):
        mc = {cam_id: self.db[cam_id] for cam_id in range(config['num_cameras'])}
        mc = copy.deepcopy(mc)
        for cam_id in range(config["num_cameras"]):
            for img_id in mc[cam_id]:
                mc[cam_id][img_id] = np.array(mc[cam_id][img_id]) * config["image_shape"]
        return mc
                    