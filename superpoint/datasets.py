"""
Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

We thank the Nanne repo https://github.com/Nanne/pytorch-NetVlad for inspiration
into the design of the dataloader
"""


import os

import torch.utils.data as data

import numpy as np
from sklearn.neighbors import NearestNeighbors


class PlaceDataset(data.Dataset):
    def __init__(
        self,
        query_file_path,
        index_file_path,
        dataset_root_dir,
        ground_truth_path,
        config,
        posDistThr=None,
    ):
        super().__init__()

        (
            self.queries,
            self.database,
            self.numQ,
            self.numDb,
            self.utmQ,
            self.utmDb,
            self.posDistThr,
        ) = (None, None, None, None, None, None, None)
        if query_file_path is not None:
            self.queries, self.numQ = self.parse_text_file(query_file_path)
        if index_file_path is not None:
            self.database, self.numDb = self.parse_text_file(index_file_path)
        if ground_truth_path is not None:
            self.utmQ, self.utmDb, self.posDistThr = self.parse_gt_file(
                ground_truth_path
            )
        if posDistThr is not None:
            self.posDistThr = posDistThr

        # if self.queries is not None:
        #     self.images = self.database + self.queries
        # else:
        #     self.images = self.database

        # self.images = [os.path.join(dataset_root_dir, image) for image in self.images]
        self.queries = [os.path.join(dataset_root_dir, image) for image in self.queries]
        self.refers = [os.path.join(dataset_root_dir, image) for image in self.database]

        self.positives = None
        self.distances = None

        self.resize = (int(config["resized_height"]), int(config["resized_width"]))

    def __len__(self):
        return len(self.images)

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)

            self.distances, self.positives = knn.radius_neighbors(
                self.utmQ, radius=self.posDistThr
            )

        return self.positives

    @staticmethod
    def parse_text_file(textfile):
        print("Parsing dataset...")

        with open(textfile, "r") as f:
            image_list = f.read().splitlines()

        # if 'robotcar' in image_list[0].lower():
        image_list = ["/".join(q_im.split("/")[2:]) for q_im in image_list]

        num_images = len(image_list)

        print("Done! Found %d images" % num_images)

        return image_list, num_images

    @staticmethod
    def parse_gt_file(gtfile):
        print("Parsing ground truth data file...")
        gtdata = np.load(gtfile)
        return gtdata["utmQ"], gtdata["utmDb"], gtdata["posDistThr"]
