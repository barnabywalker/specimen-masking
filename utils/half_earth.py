# Functions for handling metadata and images from the Half Earth dataset
import os
import numpy as np
import warnings

from collections import defaultdict
from typing import Tuple, Dict, List, Optional, Mapping, Iterable, Union

from utils.downloads import _download_kaggle, _check_exists, _extract_archive

LEVELS = {
    "order": 5,
    "family": 4,
    "genus": 3,
    "species": 2,
    "name": 1,
    "image_id": 0
}


def download_halfearth(root_dir: str) -> None:
    """Download the half earth dataset and extract it.

    Args:
        root_dir: path to a folder to download and extract the dataset into.
    """
    _check_exists(root_dir)

    base_root = os.path.dirname(root_dir)
    archive = _download_kaggle("herbarium-2021-fgvc8", base_root)
    _extract_archive(archive, base_root, remove_finished=True)


def check_levels(sample_level: str, weight_level: str) -> Tuple[str, str]:
    """Check sampling and weighting levels so they make sense.

    Args:
        sample_level: name of the hierarchical level to group sampling at, e.g. 'family'
        weight_level: name of the hierarchical level to calculate equal probability weights
                      for, e.g. 'species' will weight sampling to make it equally likely to
                      draw an image from any species.
    
    Returns:
        The sample_level and weight_level, possibly altered after a sense-check.

    Raises:
        ValueError: raises an exception if a lower value in the hierarchy is selected for weight_level
                    than sample_value.
    """
    if LEVELS[sample_level] < LEVELS[weight_level]:
        raise ValueError(f"{weight_level} is above {sample_level} in the taxonomic hierarchy, cannot calculate sensible sampling weights.")
    elif (LEVELS[sample_level] == LEVELS[weight_level]) and (LEVELS[sample_level] > 0):
        warnings.warn("sample level and weight level are the same, setting weighting to equal prob of selecting each image")
        weight_level = "image_id"

    return sample_level, weight_level


def extract_categories(metadata: Mapping[str, Iterable]) -> List[Dict[str, Union[str, int]]]:
    """Complete the taxonomic information in the metadata categories.

    Args:
        metadata: the metadata JSON file for the Half-Earth dataset, loaded as a dict.

    Returns:
        A list with a dictionary specifying the full taxonomy for each category in the metadata.
    """
    return [{
        "genus": c["name"].split()[0], 
        "species": " ".join(c["name"].split()[:2]), 
        **c
    } for c in metadata["categories"]]


def extract_annotations(
        metadata: Mapping[str, Iterable], 
        categories: Optional[Iterable[Mapping[str, Union[str, int]]]] = None
    ) -> List[Dict[str, Union[str, int]]]:
    """Extract annotation data from the metadata, and optionally insert taxonomic info about categories.

    Args:
        metadata: the metadata JSON file for the Half-Earth dataset, loaded as a dict.
        categories: the taxonomic info extracted using `extract_categories`.

    Returns:
        The annotations extracted from the metadata, optionally with the full taxonomic category
        information inserted.
    """
    annotations = metadata["annotations"]

    if categories is not None:
        annotations = [{
            "family": categories[ann['category_id']]['family'],
            "genus": categories[ann['category_id']]['genus'],
            "species": categories[ann['category_id']]['species'],
            "name": categories[ann['category_id']]['family'],
            **ann 
        } for ann in annotations]

    return annotations


def compile_names(
        annotations: Iterable[Mapping], 
        sample_level: str, 
        weight_level: str
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Compile dictionaries mapping names of the sampling and weighting categories to 
    annotation indices.

    Args:
        annotations: the annotations extracted from the Half-Earth metadata by `extract_annotations`.
        sample_level: name of the hierarchical level to group sampling at.
        weight_level: name of the hierarchical level to calculate equal probability weights
                      for.

    Returns:
        Dictionaries mapping the names of the sample-level and weight-level categories to
        the indices of all annotations labelled with those categories.
    """
    sample_names = defaultdict(list)
    weight_names = defaultdict(list)

    for i, ann in enumerate(annotations):
        sample_names[ann[sample_level]].append(i)
        weight_names[ann[weight_level]].append(i)

    return sample_names, weight_names


def sample_images(
        annotations: Iterable[Mapping], 
        n: int, 
        sample_level: str = "image_id", 
        weight_level: str = "image_id", 
        seed: Optional[int] = None
    ) -> List[Dict[str, Union[str, int]]]:
    """Draw a sample image information from a list of image annotations, selecting `n` images from each
    sample-level grouping or `n` images from the total ungrouped dataset, weighted to draw image info for 
    images from each weight-level grouping with equal probability.

    Args:
        annotations: the annotations extracted from the Half-Earth metadata by `extract_annotations`.
        n: the number of images to sample (per-group if `sample_level` is higher than 'image_id')
        sample_level: name of the hierarchical level to group sampling at.
        weight_level: name of the hierarchical level to calculate equal probability weights
                      for.
        seed: random seed, set for reproducibility from random sampler.

    Returns:
        a list of the annotations for the images in the sample
    """
    if LEVELS[sample_level] > 0:
        sample_idx = sample_grouped(annotations, n, sample_level=sample_level, weight_level=weight_level, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(annotations), n, replace=False)

    return [annotations[i] for i in sample_idx]


def sample_grouped(
        annotations: Iterable[Mapping], 
        n: int, 
        sample_level: str = "image_id", 
        weight_level: str = "image_id", 
        seed: Optional[int] = None
    ) -> List[Dict[str, Union[str, int]]]:
    """Draw a sample image information from a list of image annotations, selecting n images from each
    sample-level grouping only.

    Args:
        annotations: the annotations extracted from the Half-Earth metadata by `extract_annotations`.
        n: the number of images to sample per-group
        sample_level: name of the hierarchical level to group sampling at.
        weight_level: name of the hierarchical level to calculate equal probability weights
                      for.
        seed: random seed, set for reproducibility from random sampler.

    Returns:
        a list of indices for the images in the sample
    """
    rng = np.random.default_rng(seed)
    sample_names, weight_names = compile_names(annotations, sample_level, weight_level)

    weight_counts = {name: len(idx) for name, idx in weight_names.items()}
    sample_idx = []
    for _, idx in sample_names.items():
        N = len(idx)
        weights = [N / (weight_counts[annotations[i][weight_level]]) for i in idx]
        weights = [w / sum(weights) for w in weights]

        n = min(n, N)
        sample = rng.choice(idx, size=n, replace=False, p=weights)
        sample_idx.extend(sample)

    return sample_idx