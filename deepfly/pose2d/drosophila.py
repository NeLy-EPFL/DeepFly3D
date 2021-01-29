from __future__ import absolute_import, print_function

import os
import time
from os.path import isfile
from pathlib import Path

import cv2
import torch
import torch.nn.parallel
import torch.optim
from progress.bar import Bar
from torch.utils.data import DataLoader

import deepfly.logger as logger
from deepfly.Camera import Camera
from deepfly.Config import config

from deepfly.os_util import get_max_img_id, read_camera_order
from deepfly.pose2d.ArgParse import create_parser
from deepfly.pose2d.DrosophilaDataset import DrosophilaDataset
from deepfly.pose2d.models.hourglass import hg
from deepfly.pose2d.utils.evaluation import AverageMeter, accuracy, mse_acc
from deepfly.pose2d.utils.misc import get_time, save_checkpoint, save_dict, to_numpy
from deepfly.pose2d.utils.imutils import save_image, drosophila_image_overlay

import numpy as np


from enum import Enum


class Mode(Enum):
    train = 0
    test = 1


class NoOutputBar(Bar):
    def __init__(self, *args, **kwargs):
        Bar.__init__(self, *args, **kwargs)

    def update(self):
        pass

    def next(self):
        pass

    def finish(self):
        pass

    def start(self):
        pass


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def load_weights(model, resume: str):
    if isfile(resume):
        logger.debug("Loading checkpoint '{}'".format(resume))
        checkpoint = (
            torch.load(resume)
            if torch.cuda.is_available()
            else torch.load(resume, map_location=torch.device("cpu"))
        )

        pretrained_dict = checkpoint["state_dict"]
        model.load_state_dict(pretrained_dict, strict=False)

        logger.debug(
            "Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"])
        )
    else:
        logger.debug("=> no checkpoint found at '{}'".format(resume))
        raise FileNotFoundError


def df3dLoss(output, target, joint_exists):
    # joint_exists = joint_exists.data.numpy()
    loss = joint_exists * ((output[0] - target) ** 2).sum(dim=(-1, -2)).sum()
    for j in range(1, len(output)):
        loss += joint_exists * ((output[j] - target) ** 2).sum(dim=(-1, -2)).sum()
    return loss.sum()


def weighted_mse_loss(inp, target, weights):
    out = (inp - target) ** 2
    out = out * weights.expand_as(out)
    loss = out.sum()

    return loss


def on_cuda(torch_var, *cuda_args, **cuda_kwargs):
    return (
        torch_var.cuda(*cuda_args, **cuda_kwargs)
        if torch.cuda.is_available()
        else torch_var
    )


def get_save_path_pred(unlabeled, output_folder, front):
    unlabeled_replace = unlabeled.replace("/", "-")
    save_path = os.path.join(
        "/{}".format(unlabeled),
        output_folder,
        "./{}preds_{}.pkl".format("front_" if front else "", unlabeled_replace),
    )
    save_path = Path(save_path)
    return save_path


def get_save_path_heatmap(unlabeled, output_folder):
    unlabeled_replace = unlabeled.replace("/", "-")
    save_path = os.path.join(
        "/{}".format(unlabeled),
        output_folder,
        "heatmap_{}.pkl".format(unlabeled_replace),
    )
    save_path = Path(save_path)
    return save_path


def get_output_path(path, output_folder):
    save_path = os.path.join("/{}".format(path), output_folder)

    return save_path


def process_folder(
    model,
    loader,
    unlabeled,
    output_folder,
    overwrite,
    num_classes,
    acc_joints,
    front=False,
):
    save_path_pred, save_path_heatmap = (
        get_save_path_pred(unlabeled, output_folder, front),
        get_save_path_heatmap(unlabeled, output_folder),
    )

    if os.path.isfile(save_path_pred) and not overwrite:
        logger.info("Prediction file exists, skipping pose estimation")
        return None, None
    elif os.path.isfile(save_path_pred) and overwrite:
        logger.info("Overwriting existing predictions")

    save_path_heatmap.parent.mkdir(exist_ok=True, parents=True)
    save_path_pred.parent.mkdir(exist_ok=True, parents=True)

    heatmap = False
    pred, heatmap, _, _, _ = step(
        loader=loader,
        model=model,
        optimizer=None,
        mode=Mode.test,
        heatmap=heatmap,
        epoch=0,
        num_classes=num_classes,
        acc_joints=acc_joints,
        unlabeled=True,
    )

    _, cid2cidread = read_camera_order(get_output_path(unlabeled, output_folder))
    cid_to_reverse = config["flip_cameras"]
    cid_read_to_reverse = [cid2cidread[cid] for cid in cid_to_reverse]

    pred = flip_pred(pred, cid_read_to_reverse)

    save_dict(pred, save_path_pred)
    return pred, heatmap


def flip_heatmap(heatmap, cid_read_to_reverse):
    for cam_id in cid_read_to_reverse:
        for img_id in range(heatmap.shape[1]):
            for j_id in range(heatmap.shape[2]):
                heatmap[cam_id, img_id, j_id, :, :] = cv2.flip(
                    heatmap[cam_id, img_id, j_id, :, :], 1
                )

    return heatmap


def flip_pred(pred, cid_read_to_reverse):
    pred[cid_read_to_reverse, :, :, 0] = 1 - pred[cid_read_to_reverse, :, :, 0]

    return pred


def create_dataloader():

    train_loader = DataLoader(
        DrosophilaDataset(
            data_folder="/home/user/Desktop/DeepFly3D/gizem_annot_train.pkl",
            train=True,
            img_res=args.img_res,
            hm_res=args.hm_res,
            augmentation=args.augmentation,
            num_classes=args.num_classes,
            output_folder=args.output_folder,
            front=True,
        ),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        DrosophilaDataset(
            data_folder="/home/user/Desktop/DeepFly3D/gizem_annot_test.pkl",
            train=False,
            img_res=args.img_res,
            hm_res=args.hm_res,
            augmentation=False,
            evaluation=True,
            num_classes=args.num_classes,
            output_folder=args.output_folder,
            front=True,
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def main(args):
    logger.debug(
        "Creating model '{}', stacks={}, blocks={}".format(
            args.arch, args.stacks, args.blocks
        )
    )
    model = hg(
        num_stacks=args.stacks,
        num_blocks=args.blocks,
        num_classes=args.num_classes,
        num_feats=args.features,
        inplanes=args.inplanes,
        init_stride=args.stride,
    )
    model = on_cuda(torch.nn.DataParallel(model))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=2, factor=0.5
    )

    if args.resume:
        load_weights(model, args.resume)

    logger.debug(
        "Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    if args.unlabeled:
        loader = DataLoader(
            DrosophilaDataset(
                data_folder=args.data_folder,
                train=False,
                img_res=args.img_res,
                hm_res=args.hm_res,
                augmentation=False,
                evaluation=True,
                unlabeled=args.unlabeled,
                num_classes=args.num_classes,
                max_img_id=min(get_max_img_id(args.unlabeled), args.max_img_id),
                output_folder=args.output_folder,
                front=args.front,
            ),
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=False,
        )
        pred, heatmap = process_folder(
            model,
            loader,
            args.unlabeled,
            args.output_folder,
            args.overwrite,
            num_classes=args.num_classes,
            acc_joints=args.acc_joints,
            front=args.front,
        )
        return pred, heatmap
    else:
        train_loader, val_loader = create_dataloader()
        lr = args.lr
        best_acc = 0
        for epoch in range(args.start_epoch, args.epochs):
            logger.debug("\nEpoch: %d | LR: %.8f" % (epoch + 1, lr))

            _, _, _, _, _ = step(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                mode=Mode.train,
                heatmap=False,
                epoch=epoch,
                num_classes=args.num_classes,
                acc_joints=args.acc_joints,
                unlabeled=False,
            )
            val_pred, _, val_loss, val_acc, val_mse = step(
                loader=val_loader,
                model=model,
                optimizer=optimizer,
                mode=Mode.test,
                heatmap=False,
                epoch=epoch,
                num_classes=args.num_classes,
                acc_joints=args.acc_joints,
                unlabeled=False,
            )
            scheduler.step(val_loss)
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "image_shape": args.img_res,
                    "heatmap_shape": args.hm_res,
                },
                val_pred,
                is_best,
                checkpoint=args.checkpoint,
            )


def step(
    loader, model, optimizer, mode, heatmap, epoch, num_classes, acc_joints, unlabeled
):
    keys_am = [
        "batch_time",
        "data_time",
        "losses",
        "acces",
        "mse",
        "body_coxa",
        "coxa_femur",
        "femur_tibia",
        "tibia_tarsus",
        "tarsus_tip",
    ]
    am = {k: AverageMeter() for k in keys_am}

    if mode == mode.train:
        model.train()
    elif mode == mode.test:
        model.eval()

    start = time.time()
    bar = (
        Bar("Processing", max=len(loader)) if logger.debug_enabled() else NoOutputBar()
    )
    bar.start()

    if unlabeled:
        predictions = np.zeros(
            shape=(
                config["num_cameras"],
                loader.dataset.greatest_image_id() + 1,
                config["num_predict"],
                2,
            ),
            dtype=np.float32,
        )  # num_cameras+1 for the mirrored camera 3
    else:
        predictions = None
    for i, (inputs, target, meta) in enumerate(loader):
        np.random.seed()
        am["data_time"].update(time.time() - start)
        input_var = torch.autograd.Variable(on_cuda(inputs))
        target_var = torch.autograd.Variable(on_cuda(target, non_blocking=True))

        output = model(input_var)
        heatmap_batch = output[-1].data.cpu()

        if unlabeled:
            for n, cam_read_id, img_id in zip(
                range(heatmap_batch.size(0)), meta["cam_read_id"], meta["pid"]
            ):
                smap = to_numpy(heatmap_batch[n, :, :, :])
                pr = Camera.hm_to_pred(smap, threshold_abs=0.0)
                predictions[cam_read_id, img_id, :] = pr

        if not unlabeled:
            # loss = df3dLoss(output, target_var, meta["joint_exists"].cuda().float())
            loss = torch.nn.functional.mse_loss(output[0], target_var)
            for o_idx in range(1, len(output)):
                loss += torch.nn.functional.mse_loss(output[o_idx], target_var)
            am["losses"].update(loss.item())
            if mode == mode.train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ## PLOT
        if i < 10 and not unlabeled:
            drosophila_img = drosophila_image_overlay(
                inputs, target_var.detach(), args.hm_res, 3, args.train_joints
            )

            folder_name = (
                meta["folder_name"][0].replace("/", "-")
                + meta["folder_name"][1].replace("/", "-")
                + meta["folder_name"][2].replace("/", "-")
            )
            # image_name = meta["image_name"][0]
            template = "./test_gt_{}_{}_{:06d}_{}_joint_{}.jpg"

            save_image(
                img_path=os.path.join(
                    args.checkpoint_image_dir,
                    template.format(
                        epoch, folder_name, meta["pid"][0], meta["cid"][0], []
                    ),
                ),
                img=drosophila_img,
            )
        ## PLOT

        acc = accuracy(heatmap_batch, target, acc_joints)
        # am["acces"].add(acc)
        if not unlabeled:
            mse_err = mse_acc(target_var.data.cpu(), heatmap_batch)
            am["mse"].update(mse_err.mean().item())
        am["batch_time"].update(time.time() - start)
        start = time.time()

        bar.suffix = "{epoch} | ({batch}/{size}) D:{data:.6f}s|B:{bt:.3f}s|L:{loss:.8f}|Acc:{acc: .4f}|Mse:{mse: .3f}".format(
            epoch=epoch,
            batch=i + 1,
            size=len(loader),
            data=am["data_time"].val,
            bt=am["batch_time"].val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=am["losses"].avg,
            acc=am["acces"].avg,
            mse=am["mse"].avg,
        )
        bar.next()

    bar.finish()
    return (predictions, heatmap, am["losses"].avg, am["acces"].avg, am["mse"].avg)


if __name__ == "__main__":
    # prepare
    import logging

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    l = logger.getLogger()
    l.addHandler(handler)
    l.setLevel(logging.DEBUG)

    parser = create_parser()
    args = parser.parse_args()

    args.train_joints = np.arange(args.num_classes)
    logger.debug(f"Training joints: {args.train_joints}")
    logger.debug(f"Acc joints: {args.acc_joints}")

    args.checkpoint = (
        args.checkpoint.replace(" ", "_").replace("(", "_").replace(")", "_")
    )
    args.checkpoint = os.path.join(
        args.checkpoint,
        get_time()
        + "_{}_{}_{}_{}_{}_{}".format(
            "predict" if args.unlabeled else "training",
            args.arch,
            args.stacks,
            args.img_res,
            args.blocks,
            args.name,
        ),
    )
    args.checkpoint = args.checkpoint.replace("__", "_").replace("--", "-")
    logger.debug("Checkpoint dir: {}".format(args.checkpoint))
    args.checkpoint_image_dir = os.path.join(args.checkpoint, "./images/")

    # create checkpoint dir and image dir
    if args.unlabeled is None:
        os.makedirs(args.checkpoint, exist_ok=True)
        os.makedirs(args.checkpoint_image_dir, exist_ok=True)

    main(args)
