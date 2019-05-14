from __future__ import print_function, absolute_import

import torch.utils.data as data

from drosoph3D.GUI.os_util import *
from drosoph3D.pose2d.utils.osutils import isfile
from drosoph3D.pose2d.utils.transforms import *

FOLDER_NAME = 0
IMAGE_NAME = 1


class Drosophila(data.Dataset):
    def __init__(self, data_folder, data_corr_folder=None, img_res=None, hm_res=None, train=True, sigma=1,
                 label_type='Gaussian', jsonfile="drosophilaimaging-export.json",
                 session_id_train_list=None, folder_train_list=None, multi_view=False, augmentation=False,
                 evaluation=False, unlabeled=None,
                 temporal=False, num_classes=skeleton.num_joints // 2, max_img_id=None):
        self.train = train
        self.data_folder = data_folder  # root image folders
        self.data_corr_folder = data_corr_folder
        self.json_file = os.path.join('../../data/', jsonfile)
        self.is_train = train  # training set or test set
        self.img_res = img_res
        self.hm_res = hm_res
        self.sigma = sigma
        self.label_type = label_type
        self.multi_view = multi_view
        self.augmentation = augmentation
        self.evaluation = evaluation
        self.unlabeled = unlabeled
        self.temporal = temporal
        self.num_classes = num_classes
        self.max_img_id = max_img_id
        self.cidread2cid = dict()

        self.session_id_train_list = session_id_train_list
        self.folder_train_list = folder_train_list
        assert (not self.evaluation or not self.augmentation)  # self eval then not augmentation
        assert (not self.unlabeled or evaluation)  # if unlabeled then evaluation

        manual_path_list = ['/data/paper',
                            '/mnt/NAS/CLC/190220_SS25469-tdTomGC6fopt/Fly1/CO2xzGG/behData_001/images/',
                            '/mnt/NAS/CLC/190227_SS25478-tdTomGC6fopt/Fly3/CO2xzGG/behData_002/images/',
                            '/mnt/NAS/CLC/190227_SS25478-tdTomGC6fopt/Fly3/CO2xzGG/behData_003/images/',
                            '/mnt/NAS/CLC/190228_SS25478-tdTomGC6fopt/Fly1/CO2xzGG/behData_003/images/',
                            '/mnt/NAS/CLC/190226_SS28382-tdTomGC6fopt/Fly1/CO2xzGG/behData_001/images/',
                            '/mnt/NAS/CLC/190403_SS31219-tdTomGC6fopt/Fly1/CO2xzGG/behData_003/images/',
                            '/mnt/NAS/CLC/190220_SS25469-tdTomGC6fopt/Fly1/CO2xzGG/behData_002/images/'
                            '/mnt/NAS/CLC/190403_SS31219-tdTomGC6fopt/Fly1/CO2xzGG/behData_002/images/',
                            '/mnt/NAS/FA/190220_Rpr_R57C10_GC6s_tdTom/Fly3/001_volume/behData/images/',
                            '/mnt/NAS/FA/190220_Rpr_R57C10_GC6s_tdTom/Fly1/001_volume/behData/images/',
                            '/mnt/NAS/FA/190220_Rpr_R57C10_GC6s_tdTom/Fly2/001_volume/behData/images/',
                            '/mnt/NAS/FA/181220_Rpr_R57C10_GC6s_tdTom/Fly5/004_coronal/behData/images/'
                            ]

        # manual_path_list = []

        '''
        if not os.path.exists(self.data_folder):
            print("{} Folder does not exists".format(self.data_folder))
            raise FileNotFoundError
        '''
        self.annotation_dict = dict()
        self.multi_view_annotation_dict = dict()

        # parse json file annotations
        if not self.unlabeled and isfile(self.json_file):
            json_data = json.load(open(self.json_file, "r"))
            for session_id in json_data.keys():
                if session_id not in self.session_id_train_list:
                    print("Ignoring session id: {}".format(session_id))
                    continue
                for folder_name in json_data[session_id]["data"].keys():
                    if folder_name not in self.folder_train_list:
                        continue
                    for image_name in json_data[session_id]["data"][folder_name].keys():
                        key = ("/data/annot/" + folder_name, image_name)
                        # for the hand annotations, it is always correct ordering
                        self.cidread2cid[key[FOLDER_NAME]] = np.arange(skeleton.num_cameras)
                        cid_read, img_id = parse_img_name(image_name)

                        try:
                            pts = json_data[session_id]["data"][folder_name][image_name]["position"]
                        except:
                            print("Cannot get annotation for key ({},{})".format(key[0], key[1]))
                            continue
                        self.annotation_dict[key] = np.array(pts)

                        # also add the mirrored version for the right legs
                        if cid_read == 3:
                            from copy import deepcopy
                            new_key = deepcopy(key)
                            new_key = (new_key[0], constr_img_name(7, img_id))
                            new_pts = np.array(pts).copy()
                            self.annotation_dict[new_key] = np.array(new_pts)

        '''
        There are three cases:
        30: 15x2 5 points in each 3 legs, on 2 sides
        32: 15 tracked points,, plus antenna on each side
        38: 15 tracked points, then 3 stripes, then one antenna
        '''
        # read the manual correction for training data
        print("Searching for manual corrections")
        n_joints = set()
        if not self.unlabeled and self.train:
            pose_corr_path_list = []
            for root in manual_path_list:
                print("Searching recursively on {} for manual correction files".format(root))
                pose_corr_path_list.extend(list(glob.glob(os.path.join(root, "./**/pose_corr*.pkl"), recursive=True)))
                print("Number of manual correction files: {}".format(len(pose_corr_path_list)))
            for path in pose_corr_path_list:
                print("Reading manual annotations from {}".format(path))
                d = pickle.load(open(path, 'rb'))
                folder_name = d['folder']
                key_folder_name = folder_name
                if folder_name not in self.cidread2cid:
                    cidread2cid, cid2cidread = read_camera_order(folder_name)
                    self.cidread2cid[key_folder_name] = cidread2cid
                for cid in range(skeleton.num_cameras):
                    for img_id, points2d in d[cid].items():
                        cid_read = self.cidread2cid[key_folder_name].tolist().index(cid)
                        key = (key_folder_name, constr_img_name(cid_read, img_id))
                        num_heatmaps = points2d.shape[0]
                        n_joints.add(num_heatmaps)

                        pts = np.zeros((2 * self.num_classes, 2), dtype=np.float)
                        if cid < 3:
                            pts[:num_heatmaps // 2, :] = points2d[:num_heatmaps // 2, :]
                        elif cid > 3:
                            pts[num_classes:num_classes + (num_heatmaps // 2), :] = \
                                points2d[num_heatmaps // 2:, :]
                        elif cid == 3:
                            pts[:num_heatmaps // 2, :] = points2d[:num_heatmaps // 2, :]

                            # then add the cameras 7
                            from copy import deepcopy
                            new_key = deepcopy(key)
                            new_key = (new_key[0], constr_img_name(7, img_id))
                            new_pts = points2d[num_heatmaps // 2:, :]
                            self.annotation_dict[new_key] = np.array(new_pts)
                        else:
                            raise NotImplementedError

                        self.annotation_dict[key] = pts

        elif self.unlabeled:
            image_folder_path = os.path.join(self.data_folder, self.unlabeled)
            cidread2cid, cid2cidread = read_camera_order(image_folder_path)
            self.cidread2cid[self.unlabeled] = cidread2cid

            for image_name_jpg in os.listdir(image_folder_path):
                if image_name_jpg.endswith(".jpg"):
                    image_name = image_name_jpg.replace(".jpg", "")
                    key = (self.unlabeled, image_name)
                    cid_read, img_id = parse_img_name(image_name)
                    if self.max_img_id is not None and img_id > self.max_img_id:
                        continue
                    self.annotation_dict[key] = np.zeros(shape=(skeleton.num_joints, 2))
                    # if the front camera, then also add the mirrored version
                    if cid_read == 3:
                        new_key = (key[0], constr_img_name(7, img_id))
                        self.annotation_dict[new_key] = np.zeros(shape=(skeleton.num_joints, 2))

        print("Number of joints: {}".format(list(n_joints)))
        # make sure data is in the folder
        for folder_name, image_name in self.annotation_dict.copy().keys():
            cid_read, img_id = parse_img_name(image_name)
            if cid_read == 7:
                continue

            image_file_pad = os.path.join(self.data_folder, folder_name.replace('_network', ''),
                                          constr_img_name(cid_read, img_id) + '.jpg')
            image_file = os.path.join(self.data_folder, folder_name.replace('_network', ''),
                                      constr_img_name(cid_read, img_id, pad=False) + '.jpg')

            if not (os.path.isfile(image_file) or os.path.isfile(image_file_pad)):
                self.annotation_dict.pop((folder_name, image_name), None)
                print("FileNotFound: {}/{} ".format(folder_name, image_name))

        # preprocess the annotations and fill the multi-view dictionary
        for k, v in self.annotation_dict.copy().items():
            cid_read, img_id = parse_img_name(k[IMAGE_NAME])
            folder_name = k[FOLDER_NAME]
            cid = self.cidread2cid[folder_name][cid_read] if cid_read != 7 else 3
            if 'annot' in k[FOLDER_NAME]:
                if cid > 3:  # then right camera
                    v[:15, :] = v[15:30:]
                    v[15] = v[35]  # 35 is the second antenna
                elif cid <= 3:
                    v[15] = v[30]  # 30 is the first antenna
                else:
                    raise NotImplementedError
                v[16:, :] = 0.0
            else:  # then manual correction
                if cid > 3:  # then right camera
                    v[:self.num_classes, :] = v[self.num_classes:, :]
                else:
                    v[:self.num_classes, :] = v[:self.num_classes, :]
                # contains only 3 legs in any of the sides
                v = v[:self.num_classes, :]  # keep only one side

            ################### FIXING ANNOTATIONS #################
            # for the particular experiment some annotations are swapped
            if "2018-05-29--19-01-50--semih" in k[FOLDER_NAME]:
                if cid_read == 0 and 750 < img_id < 921:
                    tmp= np.copy(v[5:10, :])
                    v[5:10, :] = v[10:15, :]
                    v[10:15, :] = tmp
            ################### FIXING ANNOTATIONS #################

            j_keep = np.arange(0, self.num_classes)  # removing first two joints from each leg
            v = v[j_keep, :]
            v = np.abs(v)  # removing low-confidence
            # make sure normalized
            assert (np.logical_or(0 <= v, v <= 1).all())

            self.annotation_dict[k] = v
            # then we will return 3 images, and each annotation is list of three, or if evaluation, we will return sorted images

        self.annotation_key = list(
            self.annotation_dict.keys())
        if self.evaluation:  # sort keys
            self.annotation_key.sort(key=lambda x: x[0] + "_" + x[1].split("_")[3] + "_" + x[1].split("_")[1])

        self.mean, self.std = self._compute_mean()

        print("Folders inside {} data: {}".format("train" if self.train else "validation",
                                                  set([k[0] for k in self.annotation_key])))
        print("Successfully imported {} Images in Drosophila Dataset".format(len(self)))

    def _compute_mean(self):
        file_path = os.path.abspath(os.path.dirname(__file__))
        meanstd_file = os.path.join(file_path ,'../../../weights/mean.pth.tar')
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            raise FileNotFoundError
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for k in self.annotation_key:
                img_path = os.path.join(self.data_folder, k[FOLDER_NAME], k[IMAGE_NAME] + ".jpg")
                img = load_image(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self)
            std /= len(self)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __get_image_path(self, folder_name, camera_id, pose_id, pad=True):
        img_path = os.path.join(self.data_folder, folder_name.replace('_network', ''),
                                constr_img_name(camera_id, pose_id, pad=pad) + ".jpg")
        return img_path

    def __getitem__(self, index, batch_mode=True, temporal=False):
        folder_name, img_name = self.annotation_key[index][FOLDER_NAME], self.annotation_key[index][IMAGE_NAME]
        cid_read, pose_id = parse_img_name(img_name)
        if cid_read == 7:
            cid = 7
            # cid2cidread
            cid_read = self.cidread2cid[folder_name].tolist().index(3)
            if 'annot' not in folder_name:
                flip = True
            else:
                flip = False
        else:
            cid = self.cidread2cid[folder_name][cid_read]
            flip = cid > 3

        try:
            img_orig = load_image(self.__get_image_path(folder_name, cid_read, pose_id))
        except FileNotFoundError:
            try:
                img_orig = load_image(
                    self.__get_image_path(folder_name, cid_read, pose_id, pad=False))
            except FileNotFoundError:
                print("Cannot read index {} {} {} {}".format(index, folder_name, cid_read, pose_id))
                return self.__getitem__(index + 1)

        pts = torch.Tensor(self.annotation_dict[self.annotation_key[index]])
        nparts = pts.size(0)
        assert (nparts == skeleton.num_joints // 2)
        joint_exists = np.zeros(shape=(nparts,), dtype=np.uint8)
        for i in range(nparts):
            # we convert to int as we cannot pass boolean from pytorch dataloader
            # as we decrease the number of joints to skeleton.num_joints during training
            joint_exists[i] = 1 if ((0.01 < pts[i][0] < 0.99) and (0.01 < pts[i][1] < 0.99) and (
                    skeleton.camera_see_joint(cid, i) or skeleton.camera_see_joint(
                cid, (i + skeleton.num_joints // 2)))) else 0
        if flip:
            img_orig = torch.from_numpy(fliplr(img_orig.numpy())).float()
            if ('paper' not in folder_name) or ('paper' in folder_name and cid == 7):
                pts = shufflelr(pts, width=img_orig.size(2), dataset='drosophila')
        img_norm = im_to_torch(scipy.misc.imresize(img_orig, self.img_res))

        # Generate ground truth heatmap
        tpts = pts.clone()
        target = torch.zeros(nparts, self.hm_res[0], self.hm_res[1])

        for i in range(nparts):
            if joint_exists[i] == 1:
                tpts = to_torch(pts * to_torch(np.array([self.hm_res[1], self.hm_res[0]])).float())
                target[i] = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)
            else:
                # to make invisible joints explicit in the training visualization
                # these values are not used to calculate loss
                target[i] = torch.ones_like(target[i])

        # augmentation
        if self.augmentation:
            img_norm = random_jitter(img_norm, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
            img_norm, target = random_rotation(img_norm, target, degrees=10)
        else:
            img_norm = img_norm

        img_norm = color_normalize(img_norm, self.mean, self.std)

        # ugly hack to make sure mirror of camera_id 3 saved as camera 7
        if cid == 7:
            cid_read = 7
        meta = {'inp': resize(img_orig, 600, 350), 'folder_name': folder_name, 'image_name': img_name,
                'index': index, 'center': 0, 'scale': 0, 'pts': pts, 'tpts': tpts, "cid": cid,
                "cam_read_id": cid_read, "pid": pose_id, "joint_exists": joint_exists}

        return img_norm, target, meta

    def greatest_image_id(self):
        return max([parse_img_name(k[1])[1] for k in self.annotation_key])

    def __len__(self):
        return len(self.annotation_key)
