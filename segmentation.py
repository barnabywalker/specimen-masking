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
    """
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
    sample_level, weight_level = check_levels(sample_level, weight_level)
    print(os.listdir(data))
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

    args = parser.parse_args()
    arg_dict = vars(args)
    func = arg_dict.pop("func")
    func(**vars(args))
        

if __name__ == "__main__":
    main()