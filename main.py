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
from utils import utils
from utils.matching import adaptive_spatial_matching, geometry_verification
from superpoint.superpoint import SuperPointFrontend
from superpoint.utils import (
    get_query_img_name,
    get_refer_img_name,
    print_and_store_result,
)

query_descriptors = []
refer_descriptors = []

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print("Warning: OpenCV 3 is not installed")

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
                        + get_refer_img_name(dataset, refer + utils.refer_index_offset),
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
                query_ori = cv2.imread(
                    query_dir
                    + get_query_img_name(dataset, query + utils.query_index_offset),
                    0,
                )
                # query_rgb = cv2.imread(query_dir + get_query_img_name(dataset, query + query_index_offset))

            except (IOError, ValueError) as e:
                query_ori = None
                print("Exception! \n \n \n \n")

            query_img = cv2.resize(
                query_ori, (resized_width, resized_height), interpolation=cv2.INTER_AREA
            )
            query_img = query_img.astype("float32") / 255.0
            if anchor_select_policy == "conv_filter":
                edges_query = cv2.Canny(query_ori, 300, 1000, apertureSize=5)
                # deresolution to 64 * 64
                edges_query = cv2.resize(
                    edges_query, (32, 32), interpolation=cv2.INTER_AREA
                )

            query_encoding_timer_start = time.time()
            if opt.model == "pre-trained":
                if anchor_select_policy == "keypoint":
                    keypoints, desc = fe.run_with_point(query_img)
                else:
                    desc = fe.run(query_img)
            else:
                if anchor_select_policy == "keypoint":
                    with torch.no_grad():
                        desc, keypoints = val_agent.run_with_point(query_img)
                else:
                    with torch.no_grad():
                        desc = val_agent.run(query_img)
            query_encoding_time += time.time() - query_encoding_timer_start

            matching_timer_starter = time.time()

            if method == "AttnPatch":
                anchors = np.array([], dtype=np.int64)

                if anchor_select_policy == "largest_score":
                    query_self_sim = np.dot(desc.transpose(), desc)
                    query_self_sim = np.sum(query_self_sim, axis=0)
                    query_self_sim = np.reshape(-query_self_sim, (32, 32))

                    for row in range(8):
                        for col in range(8):
                            pos = np.argmin(
                                query_self_sim[
                                    (4 * row) : (4 * (row + 1)),
                                    (4 * col) : (4 * (col + 1)),
                                ]
                            )
                            tmp_anchor = np.reshape(
                                utils.idx_table[
                                    (4 * row) : (4 * (row + 1)),
                                    (4 * col) : (4 * (col + 1)),
                                ],
                                -1,
                            )[pos]
                            anchors = np.append(anchors, tmp_anchor)
                elif anchor_select_policy == "random":
                    for row in range(8):
                        for col in range(8):
                            tmp_anchor = np.reshape(
                                utils.idx_table[
                                    (4 * row) : (4 * (row + 1)),
                                    (4 * col) : (4 * (col + 1)),
                                ],
                                -1,
                            )[np.random.randint(0, 16)]
                            anchors = np.append(anchors, tmp_anchor)
                elif anchor_select_policy == "conv_filter":
                    # randomly select 64 points from edges_query where the value is not 0
                    edges_query = np.reshape(edges_query, -1)
                    filtered_args = np.argwhere(edges_query != 0)
                    filtered_args = np.reshape(filtered_args, -1)
                    if len(filtered_args) > 64:
                        anchors = np.random.choice(filtered_args, 64, replace=False)
                    else:
                        anchors = np.random.choice(range(0, 32 * 32), 64, replace=False)
                elif anchor_select_policy == "keypoint":
                    keypoints = keypoints[:2, :]
                    keypoints = keypoints.transpose()
                    keypoints = [[item // 32 for item in subl] for subl in keypoints]
                    keypoints = [
                        list(t) for t in set(tuple(element) for element in keypoints)
                    ]
                    anchors = np.array(
                        [
                            utils.idx_table[int(item[0]), int(item[1])]
                            for item in keypoints
                        ]
                    )
                else:
                    raise ValueError(
                        "anchor_select_policy for AttnPatch should be one of [largest_score, random, conv_filter, keypoint]"
                    )

                similarity.append(
                    adaptive_spatial_matching(desc, refer_descriptors, anchors)
                )
            elif method == "FullGeomVeri":
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
            utils.params,
            20,
        )
