#' training script for fastai based UNet model, for specimen segmentation.
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import wandb

from fastai.callback.training import MixedPrecision
from fastai.callback.wandb import *
from fastai.data.transforms import get_image_files

from pathlib import Path
from plantcv import plantcv as pcv
from PIL import Image

from argparse import ArgumentParser
from utils.half_earth import LEVELS, check_levels, download_halfearth, extract_annotations, extract_categories, sample_images
from utils.ferns import download_ferns, process_fern_masks
from utils.segmentation import setup_learner, create_dls, create_splitter, save_tensor_img

from typing import Optional
from tqdm import tqdm


def clear_pyplot_memory() -> None:
    """Utility to clear pyplot memory completely.
    """
    plt.clf()
    plt.cla()
    plt.close()

def process_masks(
        maskdir: str, 
        outdir: str, 
        kernel: int = 5, 
        iters: int = 1
    ) -> None:
    """Post-process predicted masks by adding dilation,
    which helps join things up a bit and ensure we get as much of the specimen as possible.

    Args:
        maskdir: the path to a folder containing the predicted masks.
        outdir: the path to a folder to save the files to.
        kernel: size of the kernel to apply during dilation - a larger kernel fills in more space.
        iters: number of iterations to run the dilation for - i think more iters fills in more space.
    
    Returns:
        Nothing, but should save the processed masks to the desired folder as a side-effect.
    """
    mask_files = [os.path.join(maskdir, f) for f in os.listdir(maskdir)]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for f in tqdm(mask_files):
        mask, path, fname = pcv.readimage(f, "native")
        processed = pcv.dilate(gray_img=mask, ksize=kernel, i=iters)
        processed_img = Image.fromarray(processed)
        processed_img.save(os.path.join(outdir, fname))

def unet(cmd: str, **kwargs) -> None:
    """Run commands for UNet training and inference.

    Args:
        cmd: which UNet function to run.
        kwargs: keyword arguments passed on from main script CLI.
    """
    if cmd == "lr_find":
        find_unet_lr(**kwargs)
    elif cmd == "train":
        train_unet(**kwargs)
    elif cmd == "predict":
        predict_unet(**kwargs)
    else:
        raise NotImplementedError(f"command {cmd} has not yet been added for this UNet model")

def find_unet_lr(
        name: str, 
        images: str, 
        masks: str, 
        metadata: Optional[str] = None, 
        save_dir: str = "output", 
        valid_pct: float = 0.8, 
        batch_size: int = 32, 
        img_size: int = 256, 
        backbone: str = "resnet34", 
        attention: bool = False,
        **kwargs
    ) -> None:
    """Produce a learning rate curve to select the best learning rate when
    training a UNet segmentation model.

    Args:
        name: an identifiable name to save the model and logs under.
        images: path to a folder containing the input images.
        masks: path to a folder containing matching ground-truth masks.
        metadata: path to a metadata file used to determine train and validation sets (e.g for Smithsonian ferns)
        save_dir: path to a folder to save outputs under.
        valid_pct: proportion of the input data to use as a held-out validation set. Not used if a metadata file is provided.
        batch_size: number of images in each batch.
        img_size: target size of images, in pixels. Expects a single number so images will be resized to squares using padding.
        backbone: name of a pretrained backbone. I think only resnet models work in this?
        attention: whether to use self-attention or not.
    """
    splitter = create_splitter(meta_path=metadata, valid_pct=valid_pct)
    dls = create_dls(images, masks, size=img_size, splitter=splitter, batch_size=batch_size)

    out_path = Path(save_dir)
    outdir = out_path/name
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model = setup_learner(dls, backbone, attention=attention, outdir=outdir)
    
    model.lr_find(show_plot=True) 
    plt.savefig(os.path.join(outdir, "lr-plot.png"))
    clear_pyplot_memory()


def train_unet(
        name: str, 
        images: str, 
        masks: str, 
        metadata: Optional[str] = None, 
        save_dir: str = "output", 
        valid_pct: float = 0.8, 
        batch_size: int = 32, 
        img_size: int = 256, 
        backbone: str = "resnet34", 
        attention: bool = False, 
        lr: float = 1e-3, 
        epochs: int = 10, 
        two_stage: bool = False,
        log_wandb: bool = False,
        **kwargs
    ) -> None:
    """Set up and train a UNet model. This uses a one-cycle learning rate scheduler.

    Args:
        name: an identifiable name to save the model and logs under.
        images: path to a folder containing the input images.
        masks: path to a folder containing matching ground-truth masks.
        metadata: path to a metadata file used to determine train and validation sets (e.g for Smithsonian ferns)
        save_dir: path to a folder to save outputs under.
        valid_pct: proportion of the input data to use as a held-out validation set. Not used if a metadata file is provided.
        batch_size: number of images in each batch.
        img_size: target size of images, in pixels. Expects a single number so images will be resized to squares using padding.
        backbone: name of a pretrained backbone. I think only resnet models work in this?
        attention: whether to use self-attention or not.
        lr: maximum learning rate.
        epochs: number of epochs to train for. If running a two-stage training, both stages will run for this number of epochs.
        two_stage: whether to unfreeze to whole pretrained model, for a second stage of training.
        log_wandb: optionally log the training process in W&B (https://wandb.ai/site).

    Returns:
        All outputs are saved under `save_dir`, so nothing is returned.
    """
    splitter = create_splitter(meta_path=metadata, valid_pct=valid_pct)
    dls = create_dls(images, masks, size=img_size, splitter=splitter, batch_size=batch_size)

    out_path = Path(save_dir)
    outdir = out_path/name
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dls.show_batch(show=True)
    plt.savefig(outdir/"segmentation-batch.png")
    clear_pyplot_memory()

    model = setup_learner(dls, backbone, attention=attention, outdir=outdir)

    callbacks = [MixedPrecision()]
    if log_wandb:
        wandb.init(project='unet-segmenter')
        callbacks.append(WandbCallback())

    model.fit_one_cycle(epochs, slice(lr), cbs=callbacks)

    if two_stage:
        lrs = slice(lr / 400, lr / 4)
        model.unfreeze()

        model.fit_one_cycle(epochs, lrs, cbs=callbacks)

    model.save(f"specimen-segmentation_{name}")


def predict_unet(
        name: str, 
        pred_dir: str,
        images: str, 
        masks: str, 
        metadata: Optional[str] = None, 
        save_dir: str = "output", 
        valid_pct: float = 0.8, 
        batch_size: int = 32, 
        img_size: int = 256, 
        backbone: str = "resnet34", 
        attention: bool = False,
        **kwargs
    ):
    """Set up and load a pretrained UNet model to make predictions with. I think this needs the original model
    to be set up in exactly the same way, including dataloaders. If I'd been clever, I'd have stored most of this
    in a config json.

    Args:
        name: the name of the model you want to load.
        pred_dir: path to a directory with images you want to generate masks for.
        images: path to the original directory with images used to train the loaded model.
        masks: path to the original directory with masks used to train the loaded model.
        metadata: path to the metadata file used to split the original dataset into train and validation sets.
        save_dir: path to a directory to save the predictions under.
        valid_pct: the original proportion of the original dataset used for validation, if a metadata file wasn't used.
        batch_size: the original batch size used when training the model.
        img_size: the target size of the predictions, must match the model being used.
        backbone: the pretrained backbone of the original model.
        attention: whether the original model uses self-attention.
    """
    splitter = create_splitter(meta_path=metadata, valid_pct=valid_pct)
    dls = create_dls(images, masks, size=img_size, splitter=splitter, batch_size=batch_size)
    
    out_path = Path(save_dir)
    outdir = out_path/name

    model = setup_learner(dls, backbone, attention=attention, outdir=outdir)
    model = model.load(f"specimen-segmentation_{name}")

    if not os.path.exists(outdir/"transformed_img"):
        os.makedirs(outdir/"transformed_img")

    if not os.path.exists(outdir/"pred_masks"):
        os.makedirs(outdir/"pred_masks")
    img_paths = get_image_files(pred_dir)
    pred_dl = model.dls.test_dl(img_paths)

    imgs_tfm, _, _, masks = model.get_preds(dl=pred_dl, with_decoded=True, with_input=True)

    for i in tqdm(range(imgs_tfm.shape[0]), desc="saving predictions"):
        save_tensor_img(imgs_tfm[i], outdir/"transformed_img"/f"{img_paths[i].stem}.png")
        save_tensor_img(masks[i], outdir/"pred_masks"/f"{img_paths[i].stem}.png")
    

def download_dataset(dataset: str, root: str) -> None:
    """Download a dataset for training or prediction.

    Args:
        dataset: the name of the dataset, one of 'ferns' or 'halfearth'.
        root: path to a directory to save the dataset under.
    """
    if dataset == "ferns":
        download_ferns(root)
        process_fern_masks(os.path.join(root, "masks", "hires_masks"), os.path.join(root, "processed_masks"))
    elif dataset == "halfearth":
        download_halfearth(root)


def sample_halfearth(
        n: int, 
        data: str, 
        out: str, 
        sample_level: str, 
        weight_level: str, 
        seed: Optional[int] = None
    ) -> None:
    """Draw a sample from the half-earth dataset (which is very big).

    Args:
        n: the number of samples to draw, per group in sample-level if above 'name'.
        data: path to the folder containing the dataset.
        out: path to save the sample under.
        sample_level: the taxonomic level to sample at.
        weight_level: the taxonomic level to weight the images at, for balanced sampling.
        seed: a random seed, to set for reproducibility.
    """
    sample_level, weight_level = check_levels(sample_level, weight_level)
    
    if not os.path.exists(os.path.join(out, "images")):
        os.makedirs(os.path.join(out, "images"))

    with open(os.path.join(data, "train", "metadata.json"), "r") as infile:
        metadata = json.load(infile)
        
    categories = extract_categories(metadata)
    annotations = extract_annotations(metadata, categories=categories)

    n_unique = len(set(c[sample_level] for c in categories)) if sample_level != "image_id" else len(annotations)

    print(f"There are {len(metadata['annotations'])} images for {len(metadata['categories'])} taxa across {n_unique} {sample_level} values")
    print(f"Sampling {int(n) * n_unique} images ({n} images from each {sample_level})")
    print(f"Weighting images for equal probability of sampling from each {weight_level}")

    sample_metadata = sample_images(
        annotations, 
        n, 
        sample_level=sample_level, 
        weight_level=weight_level, 
        seed=seed
    )

    with open(os.path.join(out, "sample-metadata.json"), "w") as outfile:
        json.dump(sample_metadata, outfile)

    sample_ids = [img["image_id"] for img in sample_metadata]
    sampled_files = []
    for dirpath, _, fnames in os.walk(os.path.join(data, "train/images")):
        sampled_files.extend([os.path.join(dirpath, fname) for fname in fnames if int(fname.split(".")[0]) in sample_ids])

    for src_path in tqdm(sampled_files, desc="copying sampled images"):
        dest_path = os.path.join(out, "images", src_path.split("/")[-1])
        shutil.copy(src_path, dest_path)


def main():
    parser = ArgumentParser(prog="segmentation")
    subparsers = parser.add_subparsers(help="sub-command menu")

    # UNet related commands
    parser_unet = subparsers.add_parser("unet", description="setup, train, and predict with a UNet model")
    parser_unet.add_argument("cmd", choices=["lr_find", "train", "predict"])
    parser_unet.add_argument("--name", default="unet", type=str, help="the name of the model")
    parser_unet.add_argument("--images", default="data/ferns/original_hires_images", type=str, 
                           help="path to training images dir")
    parser_unet.add_argument("--masks", default="output/ferns/processed-masks", type=str, 
                           help="path to training images dir")
    parser_unet.add_argument("--pred_dir", default=None, type=str, help="path to directory with prediction inputs")
    parser_unet.add_argument("--metadata", default=None, type=str, help="path to the metadata file for the dataset")
    parser_unet.add_argument("--save_dir", default="output", type=str, help="path to directory to save outputs under")
    parser_unet.add_argument("--valid_pct", default=0.2, type=float)
    parser_unet.add_argument("--batch_size", default=32, type=int)
    parser_unet.add_argument("--img_size", default=256, type=int)
    parser_unet.add_argument("--backbone", default="resnet34", type=str, choices=["resnet18", "resnet34"])
    parser_unet.add_argument('--self-attention', dest='attention', action='store_true')
    parser_unet.set_defaults(attention=False)
    parser_unet.add_argument("--lr", default=1e-3, type=float)
    parser_unet.add_argument("--epochs", default=10, type=int)
    parser_unet.add_argument("--two-stage", dest="two_stage", action="store_true")
    parser_unet.set_defaults(two_stage=False)
    parser_unet.add_argument("--log_wandb", dest='log_wandb', action='store_true', help="use wandb for experiment logging")
    parser_unet.set_defaults(log_wandb=False)
    parser_unet.set_defaults(func=unet)
    
    # sampling the half earth dataset for segmentation mask training
    parser_he = subparsers.add_parser("sample_halfearth", description="sample the half-earth dataset for a training set")
    parser_he.add_argument("n", default=1, type=int, help="number of images (per group if sample-level is above image)")
    parser_he.add_argument("--data", default="data/half-earth/", type=str,
                        help="path to dataset directory")
    parser_he.add_argument("--out", default="./", type=str, help="path and name of json file to save sample data to")
    parser_he.add_argument("--sample_level", default="family", choices=list(LEVELS.keys()),
                        help="taxonomic level for selecting images")
    parser_he.add_argument("--weight_level", default="name", choices=list(LEVELS.keys()),
                        help="level for calculating equal probability weights")
    parser_he.add_argument("--seed", default=None, type=int)
    parser_he.set_defaults(func=sample_halfearth)
    
    # downloading relevant datasets
    parser_dl = subparsers.add_parser("download", description="download files for a dataset")
    parser_dl.add_argument("dataset", choices=["ferns", "halfearth"])
    parser_dl.add_argument("--root", default="data/", type=str)
    parser_dl.set_defaults(func=download_dataset)

    # post-processing masks
    parser_proc = subparsers.add_parser("postprocess", description="post-process predicted masks")
    parser_proc.add_argument("maskdir", type=str)
    parser_proc.add_argument("--outdir", default="output", type=str)
    parser_proc.add_argument("kernel", default=5, type=int)
    parser_proc.add_argument("iters", default=1, type=int)
    parser_proc.set_defaults(func=process_masks)

    args = parser.parse_args()
    arg_dict = vars(args)
    func = arg_dict.pop("func")
    func(**vars(args))
        

if __name__ == "__main__":
    main()