# Functions to help with image and mask processing
import numpy as np
import os

from plantcv import plantcv as pcv
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple, Iterable

from utils.downloads import download_tarfile, _download_file, _check_exists

FERN_URLS = {
    "imgs": "https://smithsonian.figshare.com/ndownloader/files/17851235",
    "masks": "https://smithsonian.figshare.com/ndownloader/files/17851277",
    "metadata": "https://smithsonian.figshare.com/ndownloader/files/21488406"
}


def binarise_mask(mask: np.ndarray, threshold: int = 50, blur: Union[int, Tuple[int, int]] = 5) -> np.ndarray:
    """Ensure a specimen mask is binary by thresholding 
    then applying a median blur.

    Args:
        mask: the image mask as a 2D numpy array of size (height, width).
        threshold: the threshold above which pixels are classed as being part of the mask (between 0 and 255).
        blur: kernel size, in pixels, of the median blur to reduce noise.
    
    Returns:
        The binarised image mask as a 2D numpy array with the same size as the input mask.
    """
    binarised = pcv.threshold.binary(mask, threshold, 255, "light")
    return pcv.median_blur(binarised, blur)


def process_fern_masks(
        maskdir: str, 
        outdir: str, 
        threshold: int = 50, 
        blur: Union[int, Tuple[int, int]] = 5
) -> None:
    """Remove grey pixels from hi-res fern masks provided by the Smithsonian, by applying a
    pixel value threshold then using a median blur to reduce noise.

    Args:
        maskdir: the path to a folder containing the hi-res mask files.
        outdir: the path to a folder to save the files to.
        threshold: the value (0 - 255) above which pixels are included in the mask.
        blur: the kernel size of the median blur.
    
    Returns:
        Nothing, but should save the processed masks to the desired folder as a side-effect.
    """
    mask_files = [os.path.join(maskdir, f) for f in os.listdir(maskdir) if f.endswith(".jpg") and not f.startswith("._")]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for f in tqdm(mask_files):
        mask, path, fname = pcv.readimage(f, "native")
        processed = binarise_mask(mask, threshold=threshold, blur=blur)
        processed_img = Image.fromarray(processed)
        processed_img.save(os.path.join(outdir, fname))


def download_ferns(
        root_dir: str
):
    """Download and extract all files associated with the dataset used to
    train the Smithsonian fern segmentation model.

    Args:
        root_dir: path to a directory to download and extract all files into.
    """

    _check_exists(root_dir)
    
    base_root = os.path.dirname(root_dir)

    img_root = os.path.join(base_root, "images")
    if not os.path.exists(img_root):
        os.makedirs(img_root)

    mask_root = os.path.join(base_root, "masks")
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)

    download_tarfile(FERN_URLS["imgs"], img_root, download_name="images.tar.gz", chunk_size=2048)
    download_tarfile(FERN_URLS["masks"], mask_root, download_name="masks.tar.gz", chunk_size=2048)
    _download_file(FERN_URLS["metadata"], base_root, download_name="metadata.tsv", chunk_size=2048)

