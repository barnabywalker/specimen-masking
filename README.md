# Herbarium specimen segmentation masks

The repository houses code to train and use a UNet-based segmentation model for producing segmentation masks of herbarium specimens.

## Motivation

This code is based on a segmentation model for [fern herbarium specimen images](https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11352), produced by the [Smithsonian Data Science Lab](https://datascience.si.edu/).

Their model was trained using `fastai v1` and they provide a [pickle of the trained model](https://smithsonian.figshare.com/articles/software/Fern_Segmentation_pickle_file/12967889) that can be used to predict masks for new specimens, using [code that they provide](https://github.com/sidatasciencelab/fern_segmentation).

I found that this model has a few limitations:
    - It works very well for masks of ferns, but tends to create voids in things like fruits or anything that's a bit reflective.
    - The images are limited to 256x256 pixels in size, which can limit the detail in the image.
    - The images a centre-cropped to acheive equal aspect ratio, which can miss parts of specimens close to the top and bottom of the specimen.

I've created this repository to attempt to overcome these problems by:
    1. Training a new UNet model using the data provided with the original fern segmentation model.
    2. Changing the model to output larger images (I've chosen 512x512 but you can change this if you re-use this code).
    3. Padding rather than cropping the images to achieve equal aspect ratio, before resizing.
    4. Adding utilities to sample images for a wider range of vascular plant species, from the Half-Earth Challenge dataset.
    5. Using the trained fern segmentation model to predict masks for these new images.
    6. Editing these masks to make ground-truth masks for the Half-Earth images.
    7. Fine-tuning the UNet model on these vascular plant masks.

## Installation

You can run the code in this repository by, first, downloading or cloning it.

The environment was fiddly to set up. You might be able to recreate it using the environment file:
```
conda create -f environment.yml
```

Otherwise, you can manually install the packages from the file. You might want to start by [installing PyTorch](https://pytorch.org/get-started/locally/).

## Recreating the Smithsonian fern segmentation model

To train a UNet using the Smithsonian ferns dataset, you can follow these steps:
```
# download the dataset
python segmentation.py download ferns --root="data/ferns/"

# run the learning-rate finder
python segmentation.py unet lr_find --name=resnet34-attention-ferns-512 --images=data/ferns/images/original_hires_images --masks=data/ferns/processed_masks --metadata="data/ferns/metadata.tsv" --img_size=512 --batch_size=4 --self-attention

# train the model
python segmentation.py unet train --name=resnet34-attention-ferns-512 --images=data/ferns/images/original_hires_images --masks=data/ferns/processed_masks --metadata="data/ferns/metadata.tsv" --img_size=512 --batch_size=4 --self-attention --epochs=10 --two-stage --log_wandb --lr=1e-3
```

I was training this model on a machine with a Tesla V100 GPU with 32GB of memory and 64GB of RAM. I had to reduce the batch size to 4 to fit the memory of the GPU. You can see the training process for this run (and previous, possibly worse, runs) in this [W&B project](https://wandb.ai/barnabywalker/unet-segmenter/runs/jyxwafb4).

## Making predictions with a trained model

You can use the code in this project to make predictions for any specimen images you have, but one interesting thing might be to make predictions on a more varied taxonomic group, edit these predicted masks to correct them, and re-finetune the model to hopefully generalise it to non-fern plant specimens.

I've set up the code here to help you do that with the [Half Earth Challenge](https://www.frontiersin.org/articles/10.3389/fpls.2021.787127/full) dataset from FGVC 8.
```
# download half-earth dataset
python segmentation.py download halfearth --root="data/halfearth/"
```

Because this a very large dataset, you might want to only use a sample of the images for creating new ground-truth masks. The code below draws a sample of a single image from each taxonomic family, to hopefully capture the whole range of morphological variation across vascular plants.

```
# sample one image from each family in the half-earth dataset so each genus in a family has an equal probability of being sampled
python segmentation.py sample_halfearth 1 --data=data/halfearth --out=output/halfearth-sample --sample_level=family --weight_level=genus --seed=42069
```

After sampling, you can then make predictions.
```
python segmentation.py unet predict --name=resnet34-attention-ferns-512 --pred_dir=output/halfearth-sample/images --images=data/ferns/images/original_hires_images --masks=data/ferns/processed_masks --metadata="data/ferns/metadata.tsv" --img_size=512 --batch_size=4 --self-attention
```

## Editing predicted masks

To help with editing predicted masks to make new ground-truth masks for fine-tuning, I've included a script to create a task in Label Studio.

Before creating the task, it is helpful to post-process predicted masks using dilation. The model tends to under-predict the area of specimens, so adding pixels at the edges of the predicted mask helps fill any holes and reduces the amount of fine-scale editing needed.
```
python segmentation.py postprocess output/resnet34-attention-ferns-512/pred_masks --outdir=output/resnet34-attention-ferns-512/processed_masks --kernel=5 --iters=1
```

To set up Label Studio, you need to make sure you've install the Python package as well as the SDK.
```
pip install label-studio
pip install label-studio-sdk
```

You can look at the [instructions for the SDK for any help](https://labelstud.io/guide/sdk.html), especially if there are errors (there probably will be some). The main thing you need to do is create an API key and edit the `.env` file. You should also edit this file to add the root folder of your project, for example, mine reads:
```
# contents of .env
LABEL_STUDIO_URL="http://localhost:8080"
# this is not a real api key
API_KEY="a9a9a91a1a1a"

LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/barnaby/specimen-masking
```

You then need to add the contents of `.env` to your environment. The best way I found to do that on Linux is:
```
export $(grep -v '^#' .env | xargs)
```

After that, start a Label Studio session.
```
label-studio start
```

In another terminal, run the setup script to create the labelling task.
```
python setup-labelling.py
```

A fair number of things are hard-coded in here, so you might need to fiddle with file paths to get the task set up correctly. But it should work if you run all the scripts the same as I have here.

## Issues

If you have any problems or questions, please create an issue! But this repository isn't under active development, so it might take a while to get an answer.