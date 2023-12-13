import numpy as np
from utils import utils
import cv2


def adaptive_spatial_matching(query_descriptor, refer_descriptors, anchors):
    scores = np.zeros(len(refer_descriptors))
    if query_descriptor is not None:
        for refer in range(len(refer_descriptors)):
            if refer_descriptors[refer + utils.refer_index_offset] is not None:
                score_matrix = np.dot(
                    query_descriptor.transpose()[anchors],
                    refer_descriptors[refer + utils.refer_index_offset],
                )
                score_max_vector = np.max(score_matrix, axis=1)
                where_max_matrix = np.argmax(score_matrix, axis=1)

                where = [
                    idx
                    for idx, val in enumerate(score_max_vector)
                    if val > utils.threshold
                ]
                query_where = anchors[where]
                refer_where = where_max_matrix[where]

                query_pos = np.array([], dtype=int)
                refer_pos = np.array([], dtype=int)

                for cnt in range(query_where.shape[0]):
                    query_pos = np.append(query_pos, query_where[cnt] + utils.pos_ptr)
                    refer_pos = np.append(refer_pos, refer_where[cnt] + utils.pos_ptr)

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
                refer_rois = refer_descriptors[refer + utils.refer_index_offset].T[
                    refer_roi
                ]

                mul_score = np.sum(np.multiply(query_rois, refer_rois), axis=1)
                select_roi_idx = np.where(mul_score > utils.threshold)
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

                query_2d_idx = utils.cache_table[query_roi]
                refer_2d_idx = utils.cache_table[refer_roi]

                if query_2d_idx.shape[0] > 3:
                    _, mask = cv2.findHomography(
                        refer_2d_idx,
                        query_2d_idx,
                        cv2.FM_RANSAC,
                        ransacReprojThreshold=utils.reproj_err,
                    )

                    inlier_index_keypoints = refer_2d_idx[mask.ravel() == 1]
                    inlier_count = inlier_index_keypoints.shape[0]
                    scores[refer] = inlier_count / query_descriptor.shape[0]

    return scores


def geometry_verification(q_desc, r_descs, total_refer_imgs):
    similarity_score = np.zeros(total_refer_imgs)
    if q_desc is not None:
        for refer in range(total_refer_imgs):
            # print('==> Query ' + str(query) + ' is matching ' + 'reference ' + str(refer))
            if r_descs[refer] is not None:
                score_matrix = np.dot(q_desc.transpose(), r_descs[refer])
                score_max_row = np.argmax(score_matrix, axis=1)
                score_max_col = np.argmax(score_matrix, axis=0)

                mutuals = np.atleast_1d(
                    np.argwhere(
                        score_max_row[score_max_col] == np.arange(len(score_max_col))
                    ).squeeze()
                )

                query_2d = mutuals
                refer_2d = score_max_col[mutuals]

                query_2d = utils.cache_table[query_2d]
                refer_2d = utils.cache_table[refer_2d]

                if query_2d.shape[0] > 3:
                    _, mask = cv2.findHomography(
                        refer_2d,
                        query_2d,
                        cv2.FM_RANSAC,
                        ransacReprojThreshold=utils.reproj_err,
                    )
                    inlier_index_keypoints = refer_2d[mask.ravel() == 1]
                    inlier_count = inlier_index_keypoints.shape[0]
                    similarity_score[refer] = inlier_count / q_desc.shape[0]

    return similarity_score
