import cv2
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA

from over_descriptor import datasets_ws


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith("module"):
        state_dict = OrderedDict(
            {k.replace("module.", ""): v for (k, v) in state_dict.items()}
        )
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(
        f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
        f"current_best_R@5 = {best_r5:.1f}"
    )
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(
            args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir
        )
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i * args.infer_batch_size >= len(pca_features):
                break
            for image in images:
                grayscale_img = (
                    0.2989 * image[0, :, :]
                    + 0.5870 * image[1, :, :]
                    + 0.1140 * image[2, :, :]
                )
                grayscale_img = np.asarray(grayscale_img.cpu(), dtype=np.float32)
                grayscale_img = cv2.resize(
                    grayscale_img, (256, 256), interpolation=cv2.INTER_LINEAR
                )
                feature = model(grayscale_img)
                features.append(feature)
            # Add one dimension to features
            features = torch.stack(features)
            pca_features[
                i * args.infer_batch_size : (i * args.infer_batch_size) + len(features)
            ] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca
