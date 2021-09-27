"""ChuNom Original and Synthetic Pages Dataset class."""
import argparse
from torch.utils.data import ConcatDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.chunom_generated_pages import ChuNomGeneratedPages
from text_recognizer.data.chunom_pages import ChuNomPages
from text_recognizer.data.chunom_synthetic_pages import ChuNomSyntheticPages


class ChuNomOriginalAndSyntheticAndGeneratedPages(BaseDataModule):
    """A concatenation of original and synthetic ChuNom pages datasets."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)

        self.chunom_pages = ChuNomPages(args)
        # self.chunom_syn_pages = ChuNomSyntheticPages(args)
        self.chunom_gen_pages = ChuNomGeneratedPages(args)

        self.dims = self.chunom_pages.dims
        self.output_dims = self.chunom_pages.output_dims
        self.mapping = self.chunom_pages.mapping
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        self.chunom_pages.prepare_data()
        # self.chunom_syn_pages.prepare_data()
        self.chunom_gen_pages.prepare_data()

    def setup(self, stage: str = None) -> None:
        self.chunom_pages.setup(stage)
        # self.chunom_syn_pages.setup(stage)
        self.chunom_gen_pages.setup(stage)

        self.data_train = ConcatDataset([self.chunom_gen_pages.data_train])
        self.data_val = ConcatDataset([self.chunom_gen_pages.data_val])
        self.data_test = ConcatDataset([self.chunom_gen_pages.data_test])

    # TODO: can pass multiple dataloaders instead of concatenation datasets
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multiple_loaders.html#multiple-training-dataloaders
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
    #     )

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "ChuNom Original and Synthetic Pages Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
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


if __name__ == "__main__":
    load_and_print_info(ChuNomOriginalAndSyntheticAndGeneratedPages)
