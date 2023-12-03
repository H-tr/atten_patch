#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import numpy as np
import time
import cv2
import yaml
import torch
import pickle

from tqdm import tqdm
from superpoint.superpoint import SuperPointFrontend
from superpoint.utils import (
    get_query_img_name,
    get_refer_img_name,
    print_and_store_result,
)

threshold = 0.55
reproj_err = 3
params = [threshold, reproj_err]

query_index_offset = 0
refer_index_offset = 0

query_descriptors = []
refer_descriptors = []

pos_ptr = np.array(
    [
        [-99, -98, -97, -96, -95, -94, -93],
        [-67, -66, -65, -64, -63, -62, -61],
        [-35, -34, -33, -32, -31, -30, -29],
        [-3, -2, -1, 0, 1, 2, 3],
        [29, 30, 31, 32, 33, 34, 35],
        [61, 62, 63, 64, 65, 66, 67],
        [93, 94, 95, 96, 97, 98, 99],
    ]
)

idx_table = np.reshape(np.array([val for val in range(0, 32 * 32)]), (32, 32))
cache_table = np.zeros((1024, 2), dtype=int)
for cnt in range(1024):
    ridx = int(cnt / 32)
    cidx = int(cnt % 32)
    cache_table[cnt] = np.array([ridx, cidx])

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print("Warning: OpenCV 3 is not installed")


def adaptive_spatial_matching(query_descriptor, refer_descriptors, anchors):
    scores = np.zeros(len(refer_descriptors))
    if query_descriptor is not None:
        for refer in range(len(refer_descriptors)):
            if refer_descriptors[refer + refer_index_offset] is not None:
                score_matrix = np.dot(
                    query_descriptor.transpose()[anchors],
                    refer_descriptors[refer + refer_index_offset],
                )
                score_max_vector = np.max(score_matrix, axis=1)
                where_max_matrix = np.argmax(score_matrix, axis=1)

                where = [
                    idx for idx, val in enumerate(score_max_vector) if val > threshold
                ]
                query_where = anchors[where]
                refer_where = where_max_matrix[where]

                query_pos = np.array([], dtype=int)
                refer_pos = np.array([], dtype=int)

                for cnt in range(query_where.shape[0]):
                    query_pos = np.append(query_pos, query_where[cnt] + pos_ptr)
                    refer_pos = np.append(refer_pos, refer_where[cnt] + pos_ptr)

                qpos_idx = np.where(query_pos >= 0)
                query_pos = query_pos[qpos_idx]
                refer_pos = refer_pos[qpos_idx]
                qpos_idx = np.where(query_pos < 1023)
                query_pos = query_pos[qpos_idx]
                refer_pos = refer_pos[qpos_idx]
                rpos_idx = np.where(refer_pos >= 0)
                query_pos = query_pos[rpos_idx]
                refer_pos = refer_pos[rpos_idx]
                rpos_idx = np.where(refer_pos < 1023)
                query_pos = query_pos[rpos_idx]
                refer_pos = refer_pos[rpos_idx]

                query_roi = np.append(query_where, query_pos)
                refer_roi = np.append(refer_where, refer_pos)

                query_rois = query_descriptor.T[query_roi]
                refer_rois = refer_descriptors[refer + refer_index_offset].T[refer_roi]

                mul_score = np.sum(np.multiply(query_rois, refer_rois), axis=1)
                select_roi_idx = np.where(mul_score > threshold)
                query_roi = query_roi[select_roi_idx]
                refer_roi = refer_roi[select_roi_idx]
                unique, unique_indices, unique_inverse, unique_counts = np.unique(
                    query_roi,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                )
                query_roi = query_roi[unique_indices]
                refer_roi = refer_roi[unique_indices]

                query_2d_idx = cache_table[query_roi]
                refer_2d_idx = cache_table[refer_roi]

                if query_2d_idx.shape[0] > 3:
                    _, mask = cv2.findHomography(
                        refer_2d_idx,
                        query_2d_idx,
                        cv2.FM_RANSAC,
                        ransacReprojThreshold=reproj_err,
                    )

                    inlier_index_keypoints = refer_2d_idx[mask.ravel() == 1]
                    inlier_count = inlier_index_keypoints.shape[0]
                    scores[refer] = inlier_count / query_descriptor.shape[0]

    return scores


def geometry_verification(q_desc, r_descs):
    similarity_score = np.zeros(total_refer_imgs)
    if q_desc is not None:
        for refer in range(total_refer_imgs):
            # print('==> Query ' + str(query) + ' is matching ' + 'reference ' + str(refer))
            if r_descs[refer] is not None:
                score_matrix = np.dot(q_desc.transpose(), r_descs[refer])
                score_max_row = np.argmax(score_matrix, axis=1)
                score_max_col = np.argmax(score_matrix, axis=0)

                mutuals = np.atleast_1d(np.argwhere(score_max_row[score_max_col] == np.arange(len(score_max_col))).squeeze())

                query_2d = mutuals
                refer_2d = score_max_col[mutuals]

                query_2d = cache_table[query_2d]
                refer_2d = cache_table[refer_2d]

                if query_2d.shape[0] > 3:
                    _, mask = cv2.findHomography(refer_2d, query_2d, cv2.FM_RANSAC, ransacReprojThreshold=reproj_err)
                    inlier_index_keypoints = refer_2d[mask.ravel() == 1]
                    inlier_count = inlier_index_keypoints.shape[0]
                    similarity_score[refer] = inlier_count / q_desc.shape[0]

    return similarity_score

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="PyTorch SuperPoint Demo.")
    parser.add_argument("--model", type=str, default="pre-trained")
    parser.add_argument("--config", type=str, default="./config/vpr_bench_config.yaml")
    parser.add_argument("--prediction_path", type=str, default=None)
    parser.add_argument("--refer_desc_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, "r") as fin:
        config = yaml.safe_load(fin)

    dataset = opt.dataset
    total_refer_imgs = config["Dataset"][opt.dataset]["total_refer_imgs"]
    total_query_imgs = config["Dataset"][opt.dataset]["total_query_imgs"]
    refer_dir = config["Dataset"]["Root"] + config["Dataset"][opt.dataset]["refer_dir"]
    query_dir = config["Dataset"]["Root"] + config["Dataset"][opt.dataset]["query_dir"]
    dataset_dir = config["Dataset"]["Root"] + config["Dataset"][opt.dataset]["path"]
    resized_width = config["resized_width"]
    resized_height = config["resized_height"]
    sp_root = config["SuperPoint"]["Root"]

    # model parameters
    weights_path = config["model"]["weights_path"]
    nms_dist = config["model"]["nms_dist"]
    conf_thresh = config["model"]["conf_thresh"]
    nn_thresh = config["model"]["nn_thresh"]
    cuda = config["model"]["cuda"]
    method = config["model"]["method"]
    anchor_select_policy = config["model"]["anchor_select_policy"]

    if opt.prediction_path is not None:
        predictions = pickle.load(open(opt.prediction_path, "rb"))
        # print_and_store_result_ii(config, total_query_imgs, predictions,
        #                           dataset_dir, sp_root, dataset, 'Patch_NetVLAD', 20)
    else:
        if opt.refer_desc_path is not None:
            refer_descriptors = pickle.load(open(opt.refer_desc_path, "rb"))
            print("==> Successfully loaded reference descriptors.")
        else:
            total_timer_start = time.time()

            print("==> Loading pre-trained network...")
            if opt.model == "pre-trained":
                # This class runs the SuperPoint network and processes its outputs.
                fe = SuperPointFrontend(
                    weights_path=weights_path,
                    nms_dist=nms_dist,
                    conf_thresh=conf_thresh,
                    nn_thresh=nn_thresh,
                    cuda=cuda,
                )
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from utils.loader import get_module

                val_model = get_module("", config["front_end_model"])
                val_agent = val_model(config["model"], device=device)
                val_agent.loadModel()
            print("==> Successfully loaded pre-trained network.")

            refer_encoding_time = 0
            print("==> Encoding References...")
            for refer in tqdm(range(total_refer_imgs), ncols=100):
                # print('==> Refer: ' + str(refer + refer_index_offset))
                try:
                    refer_img = cv2.imread(
                        refer_dir
                        + get_refer_img_name(dataset, refer + refer_index_offset),
                        0,
                    )
                    # refer_rgb = cv2.imread(refer_dir + get_refer_img_name(dataset, refer+refer_index_offset))

                except (IOError, ValueError) as e:
                    refer_img = None
                    print("Exception! \n \n \n \n")

                refer_img = cv2.resize(
                    refer_img,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
                refer_img = refer_img.astype("float32") / 255.0

                refer_encoding_timer_start = time.time()
                if opt.model == "pre-trained":
                    desc = fe.run(refer_img)
                else:
                    with torch.no_grad():
                        desc = val_agent.run(refer_img)

                refer_descriptors.append(desc)
                refer_encoding_time += time.time() - refer_encoding_timer_start

        query_encoding_time = 0
        matching_time = 0
        similarity = []
        tmp_query_rois = []
        print("\n==> Matching...")
        for query in tqdm(range(total_query_imgs), ncols=100):
            # print('==> Query: ' + str(query + query_index_offset))
            try:
                query_img = cv2.imread(
                    query_dir + get_query_img_name(dataset, query + query_index_offset),
                    0,
                )
                # query_rgb = cv2.imread(query_dir + get_query_img_name(dataset, query + query_index_offset))

            except (IOError, ValueError) as e:
                query_img = None
                print("Exception! \n \n \n \n")

            query_img = cv2.resize(
                query_img, (resized_width, resized_height), interpolation=cv2.INTER_AREA
            )
            query_img = query_img.astype("float32") / 255.0

            # query_rgb = cv2.resize(query_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            # query_rgb = (query_rgb.astype('float32') / 255.)

            query_encoding_timer_start = time.time()
            if opt.model == "pre-trained":
                desc = fe.run(query_img)
            else:
                with torch.no_grad():
                    desc = val_agent.run(query_img)
            query_encoding_time += time.time() - query_encoding_timer_start

            matching_timer_starter = time.time()
            
            if method == "AttnPatch":
                print("==> method: AttnPatch")
                anchors = np.array([], dtype=np.int64)
                
                if anchor_select_policy == "largest_score":
                    print("==> anchor_select_policy: largest_score")
                    query_self_sim = np.dot(desc.transpose(), desc)
                    query_self_sim = np.sum(query_self_sim, axis=0)
                    query_self_sim = np.reshape(-query_self_sim, (32, 32))

                    for row in range(8):
                        for col in range(8):
                            pos = np.argmin(
                                query_self_sim[
                                    (4 * row) : (4 * (row + 1)), (4 * col) : (4 * (col + 1))
                                ]
                            )
                            tmp_anchor = np.reshape(
                                idx_table[
                                    (4 * row) : (4 * (row + 1)), (4 * col) : (4 * (col + 1))
                                ],
                                -1,
                            )[pos]
                            anchors = np.append(anchors, tmp_anchor)
                elif anchor_select_policy == "random":
                    print("==> anchor_select_policy: random")
                    for row in range(8):
                        for col in range(8):
                            tmp_anchor = np.reshape(
                                idx_table[
                                    (4 * row) : (4 * (row + 1)), (4 * col) : (4 * (col + 1))
                                ],
                                -1,
                            )[np.random.randint(0, 16)]
                            anchors = np.append(anchors, tmp_anchor)
                elif anchor_select_policy == "conv_filter":
                    print("==> anchor_select_policy: conv_filter")
                    pass
                else:
                    raise ValueError(
                        "anchor_select_policy should be one of [largest_score, random, conv_filter]"
                    )

                similarity.append(
                    adaptive_spatial_matching(desc, refer_descriptors, anchors)
                )
            elif method == "FullGeomVeri":
                print("==> method: FullGeomVeri")
                similarity.append(geometry_verification(desc, refer_descriptors))
            else:
                raise ValueError("method should be one of [AttnPatch, FullGeomVeri]")
            matching_time += time.time() - matching_timer_starter

        total_time = time.time() - total_timer_start

        print_and_store_result(
            config,
            opt,
            total_time,
            refer_encoding_time,
            matching_time,
            similarity,
            method,
            anchor_select_policy,
            params,
            20,
        )
