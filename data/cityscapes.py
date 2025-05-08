import numpy as np
import os

try:
    import cityscapesscripts.helpers.labels as CSLabels
except:
    pass

from pathlib import Path
from base import BaseMMSeg
import utils

CITYSCAPES_CONFIG_PATH = Path(__file__).parent / "config" / "cityscapes.py"
CITYSCAPES_CATS_PATH = Path(__file__).parent / "config" / "cityscapes.yml"


class CityscapesDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(image_size, crop_size, split, CITYSCAPES_CONFIG_PATH, **kwargs)
        self.names, self.colors = utils.dataset_cat_description(CITYSCAPES_CATS_PATH)
        self.n_cls = 19
        self.ignore_label = 255
        self.reduce_zero_label = False

    def update_default_config(self, config):

        path = Path(__file__).parent / "dataset"
        config.data_root = path

        config.data[self.split]["data_root"] = path
        config = super().update_default_config(config)

        return config

    def test_post_process(self, labels):
        labels_copy = np.copy(labels)
        cats = np.unique(labels_copy)
        for cat in cats:
            labels_copy[labels == cat] = CSLabels.trainId2label[cat].id
        return labels_copy

if __name__ == "__main__":
    dataset = CityscapesDataset(1024, 512, "train", normalization="vit")
    print(dataset[1])