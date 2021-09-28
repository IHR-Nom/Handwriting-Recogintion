"""ChuNom Synthetic Pages Dataset class."""
import json
import math
import os
import random
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.chunom_pages import (
    ChuNomPages,
    get_dataset_properties,
    get_transform,
    NEW_LINE_TOKEN,
    TAB_TOKEN,
    IMAGE_WIDTH,
    IMAGE_HEIGHT, resize_image, ESSENTIALS_FILENAME, MAX_LABEL_LENGTH,
)
from text_recognizer.data.iam_lines import save_images_and_labels, load_line_crops_and_labels
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed_04" / "chunom_synthetic_pages"
RAW_DATA_DIRNAME = "/data1/hong/datasets/chunom/handwritten/patches"
ORIGINAL_PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed_03" / "chunom_pages"


class ChuNomSyntheticPages(ChuNomPages):
    """
    ChuNom Handwriting database synthetic pages.
    """

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Prepare ChuNom lines such that they can be used to generate synthetic pages dataset in setup().
        """
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("ChuNomSyntheticPages.prepare_data: preparing ChuNom lines for synthetic ChuNom pages creation...")
        print("Getting ChuNom lines and loading labels...")

        all_crops, all_labels = get_all_crops_and_labels()
        train_crops, train_labels = get_patch_crops_and_labels_by_group(all_crops, all_labels, "train")
        val_crops, val_labels = get_patch_crops_and_labels_by_group(all_crops, all_labels, "val")
        test_crops, test_labels = get_patch_crops_and_labels_by_group(all_crops, all_labels, "test")

        print(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}...")
        save_images_and_labels(crops=train_crops, labels=train_labels, split="train",
                               data_dirname=PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops=val_crops, labels=val_labels, split="val", data_dirname=PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops=test_crops, labels=test_labels, split="test", data_dirname=PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        print(f"ChuNomSyntheticPages.setup({stage}): Loading train and val ChuNom patch regions ...")

        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = load_line_crops_and_labels(split, PROCESSED_DATA_DIRNAME)
            X, page_labels = generate_synthetic_pages(patch_crops=crops, patch_labels=labels)
            Y = convert_strings_to_labels(strings=page_labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            transform = get_transform(image_shape=self.dims[1:], augment=augment)  # type: ignore
            return BaseDataset(X, Y, transform=transform)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", augment=self.augment)
            self.data_val = _load_dataset(split="val", augment=self.augment)
        self.data_test = _load_dataset(split="test", augment=False)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "ChuNom Synthetic Pages Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)},  {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def get_all_crops_and_labels():
    all_crops = {}
    all_labels = {}
    # Combine train and val patch sets
    for inner_split in ["train", "val"]:
        for file_name in json.load(open(os.path.join(RAW_DATA_DIRNAME, inner_split + ".json"))):
            image_name = file_name.replace(".json", ".jpg")
            with Image.open(os.path.join(RAW_DATA_DIRNAME, image_name)) as image:
                image = ImageOps.grayscale(image)
                image = ImageOps.invert(image)
                image = image.rotate(90, fillcolor=(0, 0, 0), expand=1)
                all_crops[os.path.basename(image_name)] = image
                image_annotation = json.load(open(os.path.join(RAW_DATA_DIRNAME, file_name)))
                label = "".join(image_annotation[0]["hn_text"])
                all_labels[os.path.basename(image_name)] = label.replace(" ", "")

    assert len(all_crops) == len(all_crops)
    return all_crops, all_labels


def get_patch_crops_and_labels_by_group(all_crops, all_labels, split):
    """Load ChuNom patches and labels."""
    crops = []
    labels = []
    # Re-splitting dataset follow the original sets
    for split_crop_name in json.load(
            open(os.path.join(ORIGINAL_PROCESSED_DATA_DIRNAME / split, "_labels.json"))).keys():
        for crop_name in all_crops.keys():
            if split_crop_name in crop_name:
                crops.append(all_crops[crop_name])
                labels.append(all_labels[crop_name])

    assert len(crops) == len(labels)
    return crops, labels


def generate_synthetic_pages(
        patch_crops: List[Image.Image], patch_labels: List[str], max_batch_size: int = 20
) -> Tuple[List[Image.Image], List[str]]:
    """Generate synthetic pages and corresponding labels by randomly joining different subsets of crops."""
    paragraph_properties = get_dataset_properties()

    indices = list(range(len(patch_labels)))
    assert (max_batch_size / 2) < paragraph_properties["num_lines"]["max"]

    batched_indices_list = []
    for i in range(0, 6):
        batched_indices_list.extend([[_] for _ in indices])

    random.shuffle(batched_indices_list)
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size // 2)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=(max_batch_size // 2) + 1, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=(max_batch_size // 2) + 1, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=(max_batch_size // 2) + 1, max_batch_size=max_batch_size)
    )

    unique, counts = np.unique([len(_) for _ in batched_indices_list], return_counts=True)
    for batch_len, count in zip(unique, counts):
        print(f"{count} samples with {batch_len} lines")

    page_crops, page_labels = [], []
    for page_indices in batched_indices_list:
        page_label = ""
        for i in range(0, len(page_indices)):
            if i % 2 != 0:
                page_label += TAB_TOKEN
            elif i % 2 == 0 and i != 0:
                page_label += NEW_LINE_TOKEN
            page_label += patch_labels[page_indices[i]]

        if len(page_label) > paragraph_properties["label_length"]["max"]:
            print("Label longer than longest label in original ChuNom Paragraphs dataset - hence dropping")
            continue

        page_crop = join_batch_crops_to_form_page([patch_crops[i] for i in page_indices])
        # max_para_shape = paragraph_properties["crop_shape"]["max"]
        # if page_crop.height > max_para_shape[0] or page_crop.width > max_para_shape[1]:
        #     print("Crop larger than largest crop in original ChuNom Paragraphs dataset - hence dropping")
        #     continue

        page_crops.append(page_crop)
        page_labels.append(page_label)

    assert len(page_crops) == len(page_labels)
    return page_crops, page_labels


def join_batch_crops_to_form_page(patch_crops: Sequence[Image.Image]) -> Image.Image:
    """Horizontally stack line crops and return a single image forming the paragraph."""
    crop_shapes = np.array([_.size[::-1] for _ in patch_crops])
    max_patch_height = crop_shapes[:, 0].max()
    max_patch_width = crop_shapes[:, 1].max()
    indent = int(max_patch_width * 0.2)
    page_height = max_patch_height * (math.ceil(len(patch_crops) / 2))
    page_width = max_patch_width * 2 + indent
    # Paste patch images into to page image
    para_image = Image.new(mode="L", size=(page_width, page_height), color=0)
    current_height = 0
    for i in range(0, len(patch_crops)):
        if i % 2 == 0:
            para_image.paste(patch_crops[i], box=(0, current_height))
        else:
            para_image.paste(patch_crops[i], box=(page_width - max_patch_width, current_height))
            current_height += max_patch_height
    # Resize image to make sure image size is smaller than page size
    width, height = para_image.size
    ratio_w = IMAGE_WIDTH / width
    ratio_h = IMAGE_HEIGHT / height
    scale = min(ratio_h, ratio_w)
    para_image = resize_image(para_image, scale)
    # Add image to the background
    image = Image.new(mode="L", size=(IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
    image.paste(para_image, box=(0, 0))
    image = ImageOps.grayscale(image)
    return image


def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int, repeat=50) -> List[List[Any]]:
    """
    Generate random batches of elements in values without replacement and return the list of all batches. Batch sizes
    can be anything between min_batch_size and max_batch_size including the end points.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    grouped_values_list = []
    for i in range(0, repeat):
        start_id = 0
        while start_id < len(shuffled_values):
            num_values = random.randint(min_batch_size, max_batch_size)
            grouped_values_list.append(shuffled_values[start_id: start_id + num_values])
            start_id += num_values
        # assert sum([len(_) for _ in grouped_values_list]) == len(values)
    return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(ChuNomSyntheticPages)

    # Test
    crops, labels = load_line_crops_and_labels("train", PROCESSED_DATA_DIRNAME)
    X, page_labels = generate_synthetic_pages(patch_crops=crops, patch_labels=labels)
    with open(ESSENTIALS_FILENAME) as f:
        essentials = json.load(f)
    mapping = list(essentials["characters"])
    assert mapping is not None
    mapping = [*mapping, NEW_LINE_TOKEN, TAB_TOKEN]
    inverse_mapping = {v: k for k, v in enumerate(mapping)}
    Y = convert_strings_to_labels(strings=page_labels, mapping=inverse_mapping, length=MAX_LABEL_LENGTH)
    index = random.randint(0, len(X))
    print(index)
    print(page_labels[index])

    print(Y[index])
    X[index].save("/data1/hong/test_syn.png")
