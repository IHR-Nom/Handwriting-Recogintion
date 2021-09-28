from .util import BaseDataset
from .base_data_module import BaseDataModule
from .mnist import MNIST

# Hide lines below until Lab 2
from .emnist import EMNIST
from .emnist_lines import EMNISTLines

# Hide lines above until Lab 2

# Hide lines below until Lab 5
from .emnist_lines2 import EMNISTLines2
from .iam_lines import IAMLines

# Hide lines above until Lab 5

# Hide lines below until Lab 7
from .iam_paragraphs import IAMParagraphs
from .iam_original_and_synthetic_paragraphs import IAMOriginalAndSyntheticParagraphs

# Hide lines above until Lab 7

# Add ChuNom dataset
from .chunom_pages import ChuNomPages
from .chunom_original_and_synthetic_pages import ChuNomOriginalAndSyntheticPages
from .chunom_original_and_synthetic_and_generated_pages import ChuNomOriginalAndSyntheticAndGeneratedPages
from .chunom_generated_pages import ChuNomGeneratedPages
from .chunom_synthetic_pages import ChuNomSyntheticPages