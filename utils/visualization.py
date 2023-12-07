import os
import cv2
import pickle
import numpy as np
from utils import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from superpoint.utils import get_query_img_name, get_refer_img_name


def read_failed_cases(path):
    with open(path, "rb") as f:
        failed_cases = pickle.load(f)
    return failed_cases


# get the images of failed cases
def get_failed_images(failed_cases, query_path, ref_path):
    query_images = []
    wrong_pred = []
    ref_images = []
    for case in failed_cases:
        # the image name is in 7 digits and in the format of .jpg
        query_images.append(os.path.join(query_path, str(case[0]).zfill(7) + ".jpg"))
        wrong_pred.append(os.path.join(query_path, str(case[1]).zfill(7) + ".jpg"))
        ref_images.append(os.path.join(ref_path, str(case[2][0]).zfill(7) + ".jpg"))
    return query_images, wrong_pred, ref_images


def display_failed_images(failed_cases, QUERY_PATH, REF_PATH):
    query_images, wrong_pred, ref_images = get_failed_images(
        failed_cases, QUERY_PATH, REF_PATH
    )

    labels = ["query image", "wrong prediction", "reference image"]

    # display the images of failed cases
    for i in range(len(query_images)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        query_image = mpimg.imread(query_images[i])
        wrong_pred_image = mpimg.imread(wrong_pred[i])
        ref_image = mpimg.imread(ref_images[i])
        axes[0].imshow(query_image)
        axes[0].set_title(labels[0])
        axes[1].imshow(wrong_pred_image)
        axes[1].set_title(labels[1])
        axes[2].imshow(ref_image)
        axes[2].set_title(labels[2])
        plt.axis("off")
        plt.show()


class DatasetComparer:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.common_queries = []
        self.different_queries = []

    def compare_datasets(self):
        # Extract the query data from both datasets
        queries1 = [item[0] for item in self.data1]
        queries2 = [item[0] for item in self.data2]

        # Find common queries
        self.common_queries = list(set(queries1).intersection(queries2))

        # Find different queries
        self.different_queries = list(set(queries1).symmetric_difference(queries2))

        print(f"The number of failed cases in common is {len(self.common_queries)}")
        print(
            f"The number of failed cases in different is {len(self.different_queries)}"
        )

    def are_wrong_predictions_same(self):
        # Do the comparison if lists are empty
        if not self.common_queries and not self.different_queries:
            self.compare_datasets()

        # Initialize a dictionary to store the wrong predictions for common queries
        wrong_preds_dict = {}

        # Populate the dictionary with common queries and their corresponding wrong predictions
        for query in self.common_queries:
            wrong_preds_dict[query] = (
                self.find_wrong_prediction(self.data1, query),
                self.find_wrong_prediction(self.data2, query),
            )

        # Check if wrong predictions are the same for common queries
        same_wrong_preds = {
            query: wrong_preds
            for query, wrong_preds in wrong_preds_dict.items()
            if wrong_preds[0] == wrong_preds[1]
        }

        return same_wrong_preds

    def find_wrong_prediction(self, data, query):
        for item in data:
            if item[0] == query:
                return item[1]
        return None

    def find_data_for_different_queries(self):
        different_queries = self.different_queries

        # Initialize a list to store data for different queries in data1
        different_data_in_data1 = []

        # Find data associated with different queries in data1
        for query in different_queries:
            for item in self.data1:
                if item[0] == query:
                    different_data_in_data1.append(item)

        return different_data_in_data1


# Attention patch visualization
def visual_atten(
    query_descriptor_in,
    refer_descriptor_in,
    anchors_in,
    idx,
    is_refer,
    QUERY_PATH,
    REF_PATH,
    different_data_atten,
):
    score_matrix = np.dot(
        query_descriptor_in.transpose()[anchors_in], refer_descriptor_in
    )
    score_max_vector = np.max(score_matrix, axis=1)
    where_max_matrix = np.argmax(score_matrix, axis=1)

    where = [idx for idx, val in enumerate(score_max_vector) if val > utils.threshold]
    query_where = anchors_in[where]
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

    query_rois = query_descriptor_in.T[query_roi]
    refer_rois = refer_descriptor_in.T[refer_roi]

    mul_score = np.sum(np.multiply(query_rois, refer_rois), axis=1)
    select_roi_idx = np.where(mul_score > utils.threshold)
    query_roi = query_roi[select_roi_idx]
    refer_roi = refer_roi[select_roi_idx]
    _, unique_indices, _, _ = np.unique(
        query_roi, return_index=True, return_inverse=True, return_counts=True
    )
    query_roi = query_roi[unique_indices]
    refer_roi = refer_roi[unique_indices]

    query_2d_idx = utils.cache_table[query_roi]
    refer_2d_idx = utils.cache_table[refer_roi]

    score = 0

    if query_2d_idx.shape[0] > 3:
        _, mask = cv2.findHomography(
            refer_2d_idx,
            query_2d_idx,
            cv2.FM_RANSAC,
            ransacReprojThreshold=utils.reproj_err,
        )

        inlier_index_keypoints = refer_2d_idx[mask.ravel() == 1]
        inlier_count = inlier_index_keypoints.shape[0]
        score = inlier_count / query_descriptor_in.shape[0]

    query_rgb = cv2.imread(
        QUERY_PATH + "/" + get_query_img_name("SPED", different_data_atten[idx][0])
    )

    if is_refer:
        refer_rgb = cv2.imread(
            REF_PATH + "/" + get_refer_img_name("SPED", different_data_atten[idx][2][0])
        )
    else:
        refer_rgb = cv2.imread(
            QUERY_PATH + "/" + get_query_img_name("SPED", different_data_atten[idx][1])
        )

    query_rgb = cv2.resize(query_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    query_rgb = query_rgb.astype("float32") / 255.0

    refer_rgb = cv2.resize(refer_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    refer_rgb = refer_rgb.astype("float32") / 255.0

    query_img_labels = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
    refer_img_labels = cv2.cvtColor(refer_rgb, cv2.COLOR_RGB2BGR)

    for cnt in range(query_roi.shape[0]):
        query_where_ = query_roi[cnt]
        refer_where_ = refer_roi[cnt]

        cv2.rectangle(
            query_img_labels,
            (int(query_where_ % 32) * 8, int(query_where_ / 32) * 8),
            ((int(query_where_ % 32) + 1) * 8, (int(query_where_ / 32) + 1) * 8),
            (0, 128, 0),
            1,
        )

        cv2.rectangle(
            refer_img_labels,
            (int(refer_where_ % 32) * 8, int(refer_where_ / 32) * 8),
            ((int(refer_where_ % 32) + 1) * 8, (int(refer_where_ / 32) + 1) * 8),
            (0, 128, 0),
            1,
        )

    return query_img_labels, refer_img_labels, score

def wait_continue(figure, timeout: float = 0, continue_key: str = ' ') -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent
    is_inline = 'inline' in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:

        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ('key_press_event', 'close_event')
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.
