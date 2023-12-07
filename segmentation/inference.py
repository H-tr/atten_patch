import os

import cv2
import mmcv
import torch
import numpy as np
from utils.visualization import wait_continue
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.apis import inference_model, init_model


class Segmentor:
    """Segmentor class for performing semantic segmentation using a pre-trained model.

    Args:
        config_file (str): Path to the model's configuration file.
        checkpoint_file (str): Path to the model's checkpoint file.
        save_dir (str, optional): Directory to save the segmentation results. Defaults to None.
        device (str, optional): Device to use for inference (e.g., "cuda:0" for GPU). Defaults to "cuda:0".

    Attributes:
        config_file (str): Path to the model's configuration file.
        checkpoint_file (str): Path to the model's checkpoint file.
        save_dir (str): Directory to save the segmentation results.
        device (str): Device to use for inference.
        model: Initialized segmentation model.

    """

    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        save_dir: str = None,
        device: str = "cuda:0",
    ) -> None:
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.save_dir = save_dir
        self.fig_show_cfg=dict(frameon=False)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.device = device
        self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)

    @staticmethod
    def ade_classes():
        """ADE20K class names for external use."""
        return [
            "wall", "building", "sky", "floor", "tree", # 0-4
            "ceiling", "road", "bed", "windowpane", "grass", # 5-9
            "cabinet", "sidewalk", "person", "earth", "door", # 10-14
            "table", "mountain", "plant", "curtain", "chair", # 15-19
            "car", "water", "painting", "sofa", "shelf", # 20-24
            "house", "sea", "mirror", "rug", "field", # 25-29
            "armchair", "seat", "fence", "desk", "rock", # 30-34
            "wardrobe", "lamp", "bathtub", "railing", "cushion", # 35-39 
            "base", "box", "column", "signboard", "chest of drawers", # 40-44
            "counter", "sand", "sink", "skyscraper", "fireplace", # 45-49
            "refrigerator", "grandstand", "path", "stairs", "runway", # 50-54
            "case", "pool table", "pillow", "screen door", "stairway", # 55-59
            "river", "bridge", "bookcase", "blind", "coffee table", # 60-64
            "toilet", "flower", "book", "hill", "bench", # 65-69
            "countertop", "stove", "palm", "kitchen island", "computer", # 70-74
            "swivel chair", "boat", "bar", "arcade machine", "hovel", # 75-79
            "bus", "towel", "light", "truck", "tower", # 80-84
            "chandelier", "awning", "streetlight", "booth", "television receiver", # 85-89
            "airplane", "dirt track", "apparel", "pole", "land", # 90-94
            "bannister", "escalator", "ottoman", "bottle", "buffet", # 95-99
            "poster", "stage", "van", "ship", "fountain", # 100-104
            "conveyer belt", "canopy", "washer", "plaything",  "swimming pool", # 105-109
            "stool", "barrel", "basket", "waterfall", "tent", # 110-114
            "bag", "minibike", "cradle", "oven", "ball", # 115-119
            "food", "step", "tank", "trade name", "microwave", # 120-124
            "pot", "animal", "bicycle", "lake", "dishwasher", # 125-129
            "screen", "blanket", "sculpture", "hood", "sconce", # 130-134
            "vase", "traffic light", "tray", "ashcan", "fan", # 135-139
            "pier", "crt screen", "plate", "monitor", "bulletin board", # 140-144
            "shower", "radiator", "glass", "clock", "flag", # 145-149
        ]
        
    @staticmethod
    def ade_palette():
        """ADE20K palette for external use."""
        return [
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]
        ]
        
    @staticmethod
    def dynamic_classes():
        """Get the list of dynamic classes for masking."""
        return [2, 12, 43, 76, 80, 83, 90, 92, 98, 99, 102, 103, 107, 108, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 122, 124, 125, 126, 127, 129, 133, 134, 135, 138, 139, 142, 147]

    def set_model(self, config_file: str, checkpoint_file: str):
        """Set the model with new configuration and checkpoint files.

        Args:
            config_file (str): Path to the new model's configuration file.
            checkpoint_file (str): Path to the new model's checkpoint file.

        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)

    def inference(self, img, display: bool = False):
        """Perform inference on an input image.

        Args:
            img: Input image for inference.
            display (bool, optional): Whether to display the segmentation results. Defaults to False.

        """
        result = inference_model(self.model, img)
        # append img name to save_dir
        out_file = os.path.join(self.save_dir, img.split("/")[-1])
        # visualize the results in a new window
        # you can change the opacity of the painted segmentation map in (0, 1].
        self.get_result(
            img, result, show=display, out_file=out_file, opacity=0.5
        )
        # test a video and show the results
    
    def get_result(
        self,
        img: str,
        result: SegDataSample,
        show: bool = False,
        out_file: str = None,
        opacity: float = 0.5
    ):
        """Mask the dynamic classes and visualize the results.

        Args:
            img (str): Path to the input image.
            result (SegDataSample): Segmentation result.
            show (bool, optional): Whether to display the segmentation results. Defaults to False.
            out_file (str, optional): Path to save the segmentation result. Defaults to None.
            opacity (float, optional): Opacity of the painted segmentation map. Defaults to 0.5.

        """
        image = mmcv.imread(img, channel_order='rgb')
        classes = Segmentor.ade_classes()
        dynamic_classes = Segmentor.dynamic_classes()
        pred_img_data = self._draw_sem_seg(
            image,
            result.pred_sem_seg,
            classes,
            dynamic_classes,
            opacity
        )
        if self.save_dir is not None:
            mmcv.imwrite(mmcv.rgb2bgr(pred_img_data), out_file)
        if show:
            self.show(pred_img_data)
        
    def _get_center_loc(self, mask: np.ndarray) -> np.ndarray:
        """Get the center coordinate of the semantic segmentation.

        Args:
            mask (np.ndarray): Semantic segmentation mask.

        Returns:
            np.ndarray: Center coordinate of the semantic segmentation.

        """
        loc = np.argwhere(mask == 1)

        loc_sort = np.array(
            sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
        y_list = loc_sort[:, 0]
        unique, _, counts = np.unique(
            y_list, return_index=True, return_counts=True)
        y_loc = unique[counts.argmax()]
        y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
        center_num = len(y_most_freq_loc) // 2
        x = y_most_freq_loc[center_num][1]
        y = y_most_freq_loc[center_num][0]
        return np.array([x, y])
        
    def _draw_sem_seg(self,
        image: np.ndarray,
        sem_seg: PixelData,
        classes: None | list,
        dynamic_classes: None | list,
        opacity: float = 0.5,
    ):
        """Draw the semantic segmentation on the input image.

        Args:
            image (np.ndarray): Input image.
            sem_seg (PixelData): Semantic segmentation result.
            classes (None | list): List of class names.
            dynamic_classes (None | list): List of dynamic classes for masking.
            opacity (float, optional): Opacity of the painted segmentation map. Defaults to 0.5.

        Returns:
            np.ndarray: Image with the painted segmentation map.

        """
        num_classes = len(classes)
        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        # select those not in dynamic classes
        ids = np.array([id for id in ids if id not in dynamic_classes])
        labels = np.array(ids, dtype=np.int64)
        palette = Segmentor.ade_palette()
        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        font = cv2.FONT_HERSHEY_SIMPLEX
        # (0,1] to change the size of the text relative to the image
        scale = 0.05
        fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
        fontColor = (255, 255, 255)
        if image.shape[0] < 300 or image.shape[1] < 300:
            thickness = 1
            rectangleThickness = 1
        else:
            thickness = 2
            rectangleThickness = 2
        lineType = 2

        if isinstance(sem_seg[0], torch.Tensor):
            masks = sem_seg[0].numpy() == labels[:, None, None]
        else:
            masks = sem_seg[0] == labels[:, None, None]
        masks = masks.astype(np.uint8)
        for mask_num in range(len(labels)):
            classes_id = labels[mask_num]
            classes_color = colors[mask_num]
            loc = self._get_center_loc(masks[mask_num])
            text = classes[classes_id]
            (label_width, label_height), baseline = cv2.getTextSize(
                text, font, fontScale, thickness)
            mask = cv2.rectangle(mask, loc,
                                    (loc[0] + label_width + baseline,
                                    loc[1] + label_height + baseline),
                                    classes_color, -1)
            mask = cv2.rectangle(mask, loc,
                                    (loc[0] + label_width + baseline,
                                    loc[1] + label_height + baseline),
                                    (0, 0, 0), rectangleThickness)
            mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                font, fontScale, fontColor, thickness,
                                lineType)
        color_seg = (image * (1 - opacity) + mask * opacity).astype(np.uint8)
        return color_seg
    
    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplot manager.

        Args:
            win_name (str): The window name.
        """
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager
        if getattr(self, 'manager', None) is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)

        try:
            self.manager.set_window_title(win_name)
        except Exception:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
            self.manager.set_window_title(win_name)
        
    def show(self,
             drawn_img: None | np.ndarray,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key: str = ' ',
             backend: str = 'matplotlib') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            is_inline = 'inline' in plt.get_backend()
            img = self.get_image() if drawn_img is None else drawn_img
            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            # Find a better way for inline to show the image
            if is_inline:
                return fig
            wait_continue(fig, timeout=wait_time, continue_key=continue_key)
        elif backend == 'cv2':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f'{id(self)}')
            cv2.setWindowTitle(f'{id(self)}', win_name)
            cv2.imshow(
                str(id(self)),
                self.get_image() if drawn_img is None else drawn_img)
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')
            