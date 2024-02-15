import math
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from over_descriptor import test
from over_descriptor import util
from over_descriptor import parser
from over_descriptor import commons
from over_descriptor import datasets_ws
from over_descriptor.model import network

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
)

#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(
    args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query
)
logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

args.features_dim = 256

#### Initialize model
model = network(
    weights_path="pretrained_models/superpoint_v1.pth",
    nms_dist=4,
    conf_thresh=0.015,
    nn_thresh=0.7,
    cuda=True,
)

model = torch.nn.DataParallel(model)

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001
    )

criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(
        args, model, strict=False
    )
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}"
    )
else:
    best_r5 = start_epoch_num = not_improved_num = 0


#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False

        triplets_dl = DataLoader(
            dataset=triplets_ds,
            num_workers=args.num_workers,
            batch_size=args.train_batch_size,
            collate_fn=datasets_ws.collate_fn,
            pin_memory=(args.device == "cuda"),
            drop_last=True,
        )

        model = model.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):

            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            # Compute features of all images (images contains queries, positives and negatives)
            features = []
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
            loss_triplet = 0

            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(
                        args.train_batch_size, args.negs_num_per_query, 3
                    ),
                    1,
                    0,
                )
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet(
                        features[queries_indexes],
                        features[positives_indexes],
                        features[negatives_indexes],
                    )

            del features
            loss_triplet /= args.train_batch_size * args.negs_num_per_query

            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet

        logging.debug(
            f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "
            + f"current batch triplet loss = {batch_loss:.4f}, "
            + f"average epoch triplet loss = {epoch_losses.mean():.4f}"
        )

    logging.info(
        f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch triplet loss = {epoch_losses.mean():.4f}"
    )

    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(
        args,
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r5": best_r5,
            "not_improved_num": not_improved_num,
        },
        is_best,
        filename="last_model.pth",
    )

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
        )
        if not_improved_num >= args.patience:
            logging.info(
                f"Performance did not improve for {not_improved_num} epochs. Stop training."
            )
            break


if __name__ == "__main__":
    logging.info(f"Best R@5: {best_r5:.1f}")
    logging.info(
        f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
    )

    #### Test best model on test set
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
        "model_state_dict"
    ]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
