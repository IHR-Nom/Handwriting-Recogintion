"""ChuNom Dataset class."""
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json

import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from text_recognizer.data import util
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, split_dataset

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed_02" / "chunom_pages"

NEW_LINE_TOKEN = "\n"
TAB_TOKEN = "\t"
TRAIN_FRAC = 0.85
VAL_FRAC = 0.1
TEST_FRAC = 0.05

IMAGE_SCALE_FACTOR = 2
FINAL_IMAGE_HEIGHT = 388
FINAL_IMAGE_WIDTH = 568
IMAGE_HEIGHT = 365
IMAGE_WIDTH = 525
MAX_LABEL_LENGTH = 200

ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "chunom_characters_01.json"
RAW_DATA_DIRNAME = "/data1/hong/datasets/chunom/handwritten/pages"


class ChuNomPages(BaseDataModule):
    """
    ChuNom Handwriting database paragraphs.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        mapping = list(essentials["characters"])
        assert mapping is not None
        self.mapping = [*mapping, NEW_LINE_TOKEN, TAB_TOKEN]
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.dims = (1, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH)  # We assert that this is correct in setup()
        self.output_dims = (MAX_LABEL_LENGTH, 1)  # We assert that this is correct in setup()

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("ChuNomPages.prepare_data: Cropping ChuNom page regions and saving them along with labels...")

        properties = {}
        crops, labels = get_page_crops_and_labels()

        train_size = round(TRAIN_FRAC * len(crops))
        val_size = round(VAL_FRAC * len(crops))

        print(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}...")
        save_crops_and_labels(crops=dict(list(crops.items())[:train_size]),
                              labels=dict(list(labels.items())[:train_size]), split="train")
        save_crops_and_labels(crops=dict(list(crops.items())[train_size:train_size + val_size]),
                              labels=dict(list(labels.items())[train_size:train_size + val_size]), split="val")
        save_crops_and_labels(crops=dict(list(crops.items())[train_size + val_size:len(crops)]),
                              labels=dict(list(labels.items())[train_size + val_size:len(crops)]), split="test")

        properties.update(
            {
                id_: {
                    "crop_shape": crops[id_].size[::-1],
                    "label_length": len(label),
                    "num_lines": _num_lines(label),
                }
                for id_, label in labels.items()
            }
        )

        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)

    def setup(self, stage: str = None) -> None:
        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = load_processed_crops_and_labels(split)
            X = [crop for crop in crops]
            Y = convert_strings_to_labels(strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            transform = get_transform(image_shape=self.dims[1:], augment=augment)  # type: ignore
            return BaseDataset(X, Y, transform=transform)

        print(f"ChuNomPages.setup({stage}): Loading ChuNom paragraph regions and lines...")
        validate_input_and_output_dimensions(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", augment=self.augment)
            self.data_val = _load_dataset(split="val", augment=self.augment)

        # if stage == "test" or stage is None:
        self.data_test = _load_dataset(split="test", augment=False)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "ChuNom Pages Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def validate_input_and_output_dimensions(
        input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2


def get_page_crops_and_labels() -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Load ChuNom page crops and labels for a given split."""
    crops = {}
    labels = {}
    page_filenames = []
    img_folders = ["Luc-Van-Tien", "tale-of-kieu"]
    for img_folder in img_folders:
        for img in os.listdir(os.path.join(RAW_DATA_DIRNAME, img_folder, "images")):
            page_filenames.append(os.path.join(RAW_DATA_DIRNAME, img_folder, "images", img))
    for page_filename in page_filenames:
        image = Image.open(page_filename)
        image = ImageOps.invert(image)
        parent_dir = util.get_parent_folder(page_filename, 2)
        image_name = os.path.relpath(page_filename, parent_dir)
        bounding_box = get_image_size(image_name, os.path.join(parent_dir, "bboxes.json"))
        if bounding_box is None:
            continue
        lines = get_image_ground_truth(image_name, os.path.join(parent_dir, "annotation.json"))
        if lines is None:
            continue
        image = image.crop(bounding_box)
        # Rotate image to easy get to labels due to chunom are written from right to left, top to bottom
        image = image.rotate(90, fillcolor=(0, 0, 0), expand=1)
        # Resize image to make sure image size is smaller than page size
        width, height = image.size
        ratio_w = IMAGE_WIDTH / width
        ratio_h = IMAGE_HEIGHT / height
        scale = min(ratio_h, ratio_w)
        image = resize_image(image, scale)
        # Add image to the background
        page_image = Image.new(mode="L", size=(IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
        page_image.paste(image, box=(0, 0))
        page_image = ImageOps.grayscale(page_image)
        crops[os.path.basename(page_filename)] = page_image
        labels[os.path.basename(page_filename)] = lines
    assert len(crops) == len(labels)
    return crops, labels


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((round(image.width * scale_factor), round(image.height * scale_factor)),
                        resample=Image.BILINEAR)


def resize_filling(image, new_size, color=None):
    n_width, n_height = new_size
    height, width = image.shape[:2]
    ratio_w = n_width / width
    ratio_h = n_height / height
    ratio = min(ratio_h, ratio_w)
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    blank_image = np.zeros((n_height, n_width, 3), np.uint8)
    if color is None:
        color = bincount_app(image)
    lower = np.array([color[0] - 20, color[1] - 20, color[2] - 20])
    upper = np.array([color[0] + 20, color[1] + 20, color[2] + 20])
    mask = cv2.inRange(image, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask != 0] = color
    blank_image[:] = color

    x_offset, y_offset = int((n_width - width) / 2), int((n_height - height) / 2)
    blank_image[y_offset:y_offset + height, x_offset:x_offset + width] = masked_image.copy()
    return blank_image


def bincount_app(a):
    image_to_array = np.array(a)
    a2D = image_to_array.reshape(-1, image_to_array.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def get_image_size(image_name, bounding_box_file):
    bbox_json = json.load(open(bounding_box_file))
    image_asset = [value for value in bbox_json["assets"].values() if value["asset"]["path"] == image_name]
    if len(image_asset) == 0:
        return None
    else:
        image_regions = image_asset[0]["regions"]
    image_points = [p["points"] for p in image_regions if p.get("tags")[0] == "Column"]
    image_points = [j for i in image_points for j in i]

    max_x = max(p.get("x") for p in image_points)
    max_y = max(p.get("y") for p in image_points)
    min_x = min(p.get("x") for p in image_points)
    min_y = min(p.get("y") for p in image_points)

    return (min_x, min_y, max_x, max_y)


def get_image_ground_truth(image_name, ground_truth_file):
    image_ground_truth = ""
    annotation_json = json.load(open(ground_truth_file))
    image_annotation = [value["annotations"] for value in annotation_json if value["img"] == image_name]

    if len(image_annotation) == 0:
        return None
    image_annotation = image_annotation[0]

    for i in range(0, len(image_annotation)):
        if i % 2 != 0:
            image_ground_truth += "\t"
        elif i % 2 == 0 and i != 0:
            image_ground_truth += "\n"
        image_ground_truth += "".join(image_annotation[i]["hn_text"])

    return image_ground_truth.replace(" ", "")


def save_crops_and_labels(crops: Dict[str, Image.Image], labels: Dict[str, str], split: str):
    """Save crops, labels and shapes of crops of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with open(_labels_filename(split), "w") as f:
        json.dump(labels, f, indent=4)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))


def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for given split."""
    with open(_labels_filename(split), "r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels


def get_transform(image_shape: Tuple[int, int], augment: bool) -> transforms.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            transforms.RandomCrop(  # random pad image to image_shape with 0
                size=image_shape, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
            ),
            transforms.ColorJitter(brightness=(0.8, 1.3))
        ]
    else:
        transforms_list = [transforms.CenterCrop(image_shape)]  # pad image to image_shape with 0
    # transforms_list.append(lambda x: np.asarray(x))
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {"min": min(_get_property_values("num_lines")), "max": max(_get_property_values("num_lines"))},
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count("\n") + 1


if __name__ == "__main__":
    load_and_print_info(ChuNomPages)
