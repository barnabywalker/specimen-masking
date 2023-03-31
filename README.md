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

