"""ChuNom Generated Pages Dataset class."""
import csv
import glob
import os
import random
from typing import Any, List, Sequence, Tuple

import numpy as np
import torchvision.transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont

from text_recognizer.data import augmentation
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.chunom_pages import (
    ChuNomPages,
    get_dataset_properties,
    get_transform,
    NEW_LINE_TOKEN,
    TAB_TOKEN, TRAIN_FRAC, VAL_FRAC, IMAGE_WIDTH, IMAGE_HEIGHT, )
from text_recognizer.data.iam_lines import save_images_and_labels, load_line_crops_and_labels
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed_01" / "chunom_generated_pages"
RAW_DATA_DIRNAME = "/data1/hong/datasets/chunom/nlp/nlp_data.csv"
FONT_DIRNAME = "/data1/hong/nom_fonts"

MAX_PATCH_HEIGHT = 250
MAX_PATCH_WIDTH = 45

aug = torchvision.transforms.Compose([
    augmentation.GridDistortion(), augmentation.ImgAugTransform(), torchvision.transforms.ToPILImage()
])


class ChuNomGeneratedPages(ChuNomPages):
    """
    ChuNom Handwriting database generated pages.
    """

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Prepare ChuNom lines such that they can be used to generate generated pages dataset in setup().
        """
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("ChuNomGeneratedPages.prepare_data: preparing ChuNom lines for generated ChuNom pages creation...")
        print("Getting ChuNom lines and loading labels...")

        crops, labels = generate_patch_crops_and_labels()

        # Splitting dataset with percentage of train-val: 9:1
        train_size = round(TRAIN_FRAC * len(crops))
        val_size = round(VAL_FRAC * len(crops))

        print(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}...")
        save_images_and_labels(crops=crops[:train_size], labels=labels[:train_size], split="train",
                               data_dirname=PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops=crops[train_size:train_size + val_size],
                               labels=labels[train_size:train_size + val_size], split="val",
                               data_dirname=PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops=crops[train_size + val_size:len(crops)],
                               labels=labels[train_size + val_size:len(crops)], split="test",
                               data_dirname=PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        print(f"ChuNomGeneratedPages.setup({stage}): Loading train and val ChuNom patch regions ...")

        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = load_line_crops_and_labels(split, PROCESSED_DATA_DIRNAME)
            X, page_labels = generate_generated_pages(patch_crops=crops, patch_labels=labels)
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
            "ChuNom Generated Pages Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)},  {len(self.data_val)},  {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def generate_patch_crops_and_labels():
    """Load ChuNom labels and generate crops."""
    crops = []
    labels = []
    max_patch_len = 8

    with open(RAW_DATA_DIRNAME, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header is not None:
            for line in reader:
                if len(crops) < 300000:
                    patches = [line[0][i: i + max_patch_len] for i in range(0, len(line[0]), max_patch_len)]
                    for patch in patches:
                        labels.append(patch)
                        # Generate patch crop for each label
                        patch_crop = generate_patch(patch)
                        patch_crop = ImageOps.grayscale(patch_crop)
                        patch_crop = patch_crop.rotate(90, fillcolor=0, expand=1)
                        crops.append(patch_crop)

    assert len(crops) == len(labels)
    return crops, labels


def generate_patch(patch_label):
    image = Image.new('RGB', (MAX_PATCH_WIDTH, MAX_PATCH_HEIGHT), (0, 0, 0))
    drawer = ImageDraw.Draw(image)
    random_font_size = random.randint(27, 29)
    random_font = random.choice(glob.glob(os.path.join(FONT_DIRNAME, '*.[o|t]tf')))
    font = ImageFont.truetype(random_font, random_font_size)
    w, h = drawer.textsize(patch_label[0], font=font)
    y = random.randint(int(h * 3 / 100), int(h * 6 / 100))

    for char in patch_label:
        drawer.text(((MAX_PATCH_WIDTH - w) / 2, y), char, font=font, align='center', fill=(184, 184, 184))
        # We need to add a small space between characters, otherwise the characters will sit just next to each other
        # And not be able to distinguish anymore
        y = y + h + 3
    return image


def generate_generated_pages(
        patch_crops: List[Image.Image], patch_labels: List[str], max_batch_size: int = 14
) -> Tuple[List[Image.Image], List[str]]:
    """Generate generated pages and corresponding labels by randomly joining different subsets of crops."""
    paragraph_properties = get_dataset_properties()

    indices = list(range(len(patch_labels)))
    assert (max_batch_size / 2) < paragraph_properties["num_lines"]["max"]

    # max_num_batches: int = 1000
    # if split == "train":
    #     max_num_batches *= TRAIN_FRAC
    # elif split == "val":
    #     max_num_batches *= VAL_FRAC
    # elif split == "test":
    #     max_num_batches *= TEST_FRAC

    # batched_indices_list = [[_] for _ in range(0, min(len(indices), int(max_num_batches)))]  # batch_size = 1, len = 4001
    # for i in range(2, max_batch_size + 1):
    #     batched_indices_list.extend(
    #         generate_random_batches(values=indices, num_values=i, max_num_batches=max_num_batches)
    #     )

    batched_indices_list = [[_] for _ in indices]
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
    # assert sorted(list(itertools.chain(*batched_indices_list))) == indices

    # batched_indices_list = [[3, 3, 3, 4, 5, 6, 7, 8, 3, 3, 3, 4, 5, 6]]
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

        if len(page_label) > 198:
            # print("Label longer than longest label in original ChuNom Paragraphs dataset - hence dropping")
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

    page_image = Image.new(mode="L", size=(IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
    current_height = 0
    for i in range(0, len(patch_crops)):
        if i % 2 == 0:
            page_image.paste(patch_crops[i], box=(0, current_height))
        else:
            page_image.paste(patch_crops[i], box=(IMAGE_WIDTH - MAX_PATCH_HEIGHT, current_height))
            current_height += MAX_PATCH_WIDTH
    page_image = aug(page_image)
    return page_image


def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int) -> List[List[Any]]:
    """
    Generate random batches of elements in values without replacement and return the list of all batches. Batch sizes
    can be anything between min_batch_size and max_batch_size including the end points.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    start_id = 0
    grouped_values_list = []
    while start_id < len(shuffled_values):
        num_values = random.randint(min_batch_size, max_batch_size)
        grouped_values_list.append(shuffled_values[start_id: start_id + num_values])
        start_id += num_values
    assert sum([len(_) for _ in grouped_values_list]) == len(values)
    return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(ChuNomGeneratedPages)
    crops, labels = load_line_crops_and_labels("train", PROCESSED_DATA_DIRNAME)
    X, page_labels = generate_generated_pages(patch_crops=crops, patch_labels=labels)
    X[0].save("test_gen.png")
    # with open(ESSENTIALS_FILENAME) as f:
    #     essentials = json.load(f)
    # mapping = list(essentials["characters"])
    # assert mapping is not None
    # mapping = [*mapping, NEW_LINE_TOKEN, TAB_TOKEN]
    # inverse_mapping = {v: k for k, v in enumerate(mapping)}
    # Y = convert_strings_to_labels(strings=page_labels, mapping=inverse_mapping, length=MAX_LABEL_LENGTH)
    # X[10000].save("test_gen.png")
    # print(page_labels[10000])
    # print(Y[10000])
