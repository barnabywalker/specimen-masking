# Functions for setting up, training, and predicting with UNet segmentation models
import pandas as pd
import os

from fastai.callback.schedule import lr_find, fit_one_cycle
from fastai.callback.progress import CSVLogger
from fastai.data.transforms import get_image_files, FuncSplitter, RandomSplitter

from fastai.vision.data import SegmentationDataLoaders
from fastai.vision.augment import aug_transforms, Resize, RandomResizedCrop, IntToFloatTensor
from fastai.vision.learner import unet_learner

from fastcore.xtras import Path

from fastai.metrics import Dice

from torchvision.models.resnet import resnet34, resnet18

from typing import Optional, Callable, Union


def create_dls(
        img_dir: str, 
        mask_dir: str, 
        size: int = 256, 
        batch_size:int = 32, 
        splitter: Optional[Callable] = None
    ) -> SegmentationDataLoaders:
    """Create a fastai segmentation dataloader for training a UNet model,
    using standard image augmentations.

    Args:
        img_dir: path to a directory with the images in.
        mask_dir: path to a directory with the ground-truth masks in
        size: target size of the images.
        batch_size: size of batches to group images into.
        splitter:
    """
    if splitter is None:
        splitter = RandomSplitter(valid_pct=0.2)
    mask_dir = Path(mask_dir)
    root = os.path.commonprefix([img_dir, mask_dir])
    dls = SegmentationDataLoaders.from_label_func(
        root,
        fnames=get_image_files(img_dir),
        label_func=lambda o: mask_dir/f"{o.stem}_mask{o.suffix}",
        splitter=splitter,
        codes=["background", "specimen"],
        item_tfms=[Resize(int(size*2), method="pad", pad_mode="zeros")],
        batch_tfms=[
            *aug_transforms(pad_mode="zeros"),
            RandomResizedCrop(size=size, min_scale=0.25, max_scale=0.9, ratio=(1, 1)),
            IntToFloatTensor(div_mask=255)
        ],
        bs=batch_size
    )

    return dls


def create_splitter(meta_path: Optional[str]=None, valid_pct: float = 0.8) -> Union[FuncSplitter, RandomSplitter]:
    """Create a fastai Splitter for splitting a dataset into training and validation sets.

    Args:
        meta_path: the path to a metadata file for the Smithsonian ferns dataset.
        valid_pct: the proportion of the dataset to use for validation, if a metadata file
                   is not specified.
    
    Returns:
        A fastai Splitter object.
    """
    if meta_path is not None:
        metadata = pd.read_csv(Path(meta_path), sep="\t")
        valid_barcodes = metadata[metadata.valid_set_equals_1 == 1].CatBarcode.values
        splitter = FuncSplitter(lambda o: int(Path(o).stem) in valid_barcodes)
    else:
        splitter = RandomSplitter(valid_pct=valid_pct)

    return splitter


def setup_learner(
        dls: Callable, 
        backbone: str, 
        attention: bool = False, 
        outdir: str = "output"
    ) -> Callable:
    """Set up a UNet learner from a segmentation dataloader and a pretrained backbone.

    Args:
        dls: a SegmentationDataLoader set up to pass images and corresponding masks to a fastai model.
        backbone: the name of a pretrained backbone, currently can only be 'resnet34' or 'resnet18'.
        attention: whether or not to use self attention in the UNet architecture.
        outdir: the path to save all outputs (performance metrics, saved models) to.

    Returns:
        A fastai unet_learner, ready for training.

    Raises:
        NotImplementedError: if a backbone that hasn't been included yet is specified.
    """
    if backbone == "resnet18":
        encoder = resnet18
    elif backbone == "resnet34":
        encoder = resnet34
    else:
        raise NotImplementedError(f"model '{backbone}' has not yet been added, please use one of ['resnet34', 'resnet18']")

    
    modeldir = outdir/"model"

    model = unet_learner(dls, encoder, metrics=Dice(), self_attention=attention, cbs=CSVLogger(),
                         path=outdir, model_dir=modeldir)
    
    return model