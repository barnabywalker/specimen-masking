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

from pathlib import Path

from argparse import ArgumentParser
from utils.half_earth import LEVELS, check_levels, download_halfearth, extract_annotations, extract_categories, sample_images
from utils.ferns import download_ferns, process_fern_masks
from utils.segmentation import setup_learner, create_dls, create_splitter

from typing import Optional
from tqdm import tqdm


def clear_pyplot_memory() -> None:
    """Utility to clear pyplot memory completely.
    """
    plt.clf()
    plt.cla()
    plt.close()


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
        self_attention: bool = False
    ) -> None:
    splitter = create_splitter(metadata=metadata, valid_pct=valid_pct)
    dls = create_dls(images, masks, size=img_size, splitter=splitter, batch_size=batch_size)

    out_path = Path(save_dir)
    outdir = out_path/name

    model = setup_learner(dls, backbone, attention=self_attention, outdir=outdir)
    
    model.lr_find(show_plot=True) 
    plt.savefig(os.path.join(save_dir, "lr-plot.png"))
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
        self_attention: bool = False, 
        lr: float = 1e-3, 
        epochs: int = 10, 
        two_stage: bool = False,
        wandb: bool = False
    ) -> None:
    splitter = create_splitter(metadata=metadata, valid_pct=valid_pct)
    dls = create_dls(images, masks, size=img_size, splitter=splitter, batch_size=batch_size)

    out_path = Path(save_dir)
    outdir = out_path/name

    dls.show_batch(show=True)
    plt.savefig(outdir/"segmentation-batch.png")
    clear_pyplot_memory()

    model = setup_learner(dls, backbone, self_attention, name, outdir)

    callbacks = [MixedPrecision()]
    if wandb:
        wandb.init(project='unet-segmenter')
        callbacks.append(WandbCallback())

    model.fit_one_cycle(epochs, slice(lr), cbs=callbacks)

    if two_stage:
        lrs = slice(lr / 400, lr / 4)
        model.unfreeze()

        model.fit_one_cycle(epochs, lrs, cbs=callbacks)

    model.show_results(show_plot=True)
    plt.savefig(outdir/"results-plot.png")
    clear_pyplot_memory()

    model.save(f"specimen-segmentation_{name}")
    
def download_dataset(dataset: str, root: str) -> None:
    if dataset == "ferns":
        download_ferns(root)
        process_fern_masks(os.path.join(root, "masks"), os.path.join(root, "processed_masks"))
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
    sample_level, weight_level = check_levels(sample_level, weight_level)

    if not os.path.exists(os.path.join(out, "images")):
        os.makedirs(os.path.join(out, "images"))

    with open(os.path.join(data, "metadata.json"), "r") as infile:
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
    for dirpath, _, fnames in os.walk(os.path.join(data, "train")):
            sampled_files.extend([os.path.join(dirpath, fname) for fname in fnames if fname.split(".")[0] in sample_ids])

    for src_path in tqdm(sampled_files, desc="copying sampled images"):
        dest_path = os.path.join(out, "images", src_path.split("/")[-1])
        shutil.copy(src_path, dest_path)


def main():
    parser = ArgumentParser(prog="segmentation")
    subparsers = parser.add_subparsers(help="sub-command menu")

    # UNet related commands
    parser_unet = subparsers.add_parser("unet", description="setup, train, and predict with a UNet model")
    parser_unet.add_argument("--name", default="unet", type=str, help="the name of the model")
    parser_unet.add_argument("--images", default="data/ferns/original_hires_images", type=str, 
                           help="path to training images dir")
    parser_unet.add_argument("--masks", default="output/ferns/processed-masks", type=str, 
                           help="path to training images dir")
    parser_unet.add_argument("--metadata", default=None, type=str, help="path to the metadata file for the dataset")
    parser_unet.add_argument("--save_dir", default="output", type=str)
    
    parser_unet.add_argument("--valid_pct", default=0.2, type=float)
    parser_unet.add_argument("--batch_size", default=32, type=int)
    parser_unet.add_argument("--img_size", default=256, type=int)
    parser_unet.add_argument("--backbone", default="resnet34", type=str, choices=["resnet18", "resnet34"])
    parser_unet.add_argument('--self-attention', dest='attention', action='store_true')
    parser_unet.set_defaults(attention=False)
    
    unet_subparsers = parser_unet.add_subparsers(help="unet sub-commands")
    ## finding unet learning rate
    parser_lr = unet_subparsers.add_parser("lr_find", description="find the best learning rate for training")
    parser_lr.set_defaults(func=find_unet_lr)
    ## training unet
    parser_tr = unet_subparsers.add_parser("train", description="train a unet model for image segmentation")
    parser_tr.add_argument("--lr", default=1e-3, type=float)
    parser_tr.add_argument("--epochs", default=10)
    parser_tr.add_argument("--two-stage", dest="two_stage", action="store_true")
    parser_tr.set_defaults(two_stage=False)
    parser_tr.add_argument("--wandb", default=False, type=bool, help="use wandb for experiment logging")
    parser_tr.set_defaults(func=train_unet)
    
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

    args = parser.parse_args()
    print(args)
    return()
    args.func(args)
        

if __name__ == "__main__":
    main()