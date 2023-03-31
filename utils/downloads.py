# Utilities for downloading files
import kaggle
import numpy as np
import os
import requests
import tarfile
import zipfile

from tqdm import tqdm
from typing import Optional

def _check_exists(root_dir: str) -> None:
    """Check if a directory exists and has files in it, raising an error if it does.

    Args:
        root_dir: the path to the directory to check.

    Raises:
        RuntimeError if the directory exists and has at least one file int it.
    """
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) > 0:
        raise RuntimeError(
            f"The directory {root_dir} already exists.",
            "To re-download or re-extract the files, please delete the directory."
        )

def download_tarfile(
    url: str,
    download_root: str,
    download_name: Optional[str] = None,
    extract_root: Optional[str] = None,
    chunk_size: Optional[int] = 128,
    remove_finished: bool = False
) -> None:
    """Download and extract a tar archive from a URL.

    Args:
        url: the url of the tar archive.
        download_root: path to a directory to save the archive into.
        download_name: the name to give the downloaded archive.
        extract_root: path to a directory to extract the archive into.
        chunk_size: the number of bytes to request at a time for a large archive file.
        remove_finished: whether to remove the downloaded archive after extracting, or not.
    """
    if extract_root is None:
        extract_root = download_root

    archive = _download_file(url, download_root, download_name=download_name, chunk_size=chunk_size)
    _extract_archive(archive, extract_root, remove_finished=remove_finished)


def _download_file(
    url: str,
    download_root: str,
    download_name: Optional[str] = None,
    chunk_size: Optional[int] = 128
) -> str:
    """Download a large file from a URL in chunks.

    Args:
        url: the URL of the file.
        download root: path to a directory to save the file into.
        download_name: a string to name the downloaded file.
        chunk_size: the number of bytes to request at once.

    Returns:
        the path to the downloaded file.
    """
    if download_name is None:
        download_name = url.split("/")[-1]

    archive = os.path.join(download_root, download_name)
    response = requests.get(url, stream=True)

    nbytes = int(response.headers["Content-Length"])
    nchunks = int(np.ceil(nbytes / chunk_size))
    with open(archive, 'wb') as urlfile:
        for chunk in tqdm(response.iter_content(chunk_size=chunk_size), total=nchunks, 
                          desc=f"downloading file from {url} to {download_root}"):
            urlfile.write(chunk)

    return archive


def _extract_archive(
    archive: str,
    extract_root: str,
    remove_finished: Optional[bool] = True
):
    """Extract an archive file and optionally delete the original archive file.
    Should work for zip and tar files.

    Args:
        archive: path to the archive file.
        extract_root: path to a directory to extract the archive into.
        remove_finished: whether to delete the original archive file or not.
    """
    print(f"Extracting file from {archive} to {extract_root}")

    if archive.endswith("zip"):
        with zipfile.ZipFile(archive, "r", compression=zipfile.ZIP_STORED) as zip:
            zip.extractall(extract_root)
    elif archive.endswith("tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)
    else:
        raise ValueError(f"{archive} not a recognised archive file.")

    if remove_finished:
        os.remove(archive)

def _download_kaggle(
    dataset_name: str,
    download_root: str
) -> None:
    """Download a dataset from Kaggle using the Kaggle API. 
    You need to store your Kaggle username and api key somewhere for this to work,
    try following https://github.com/Kaggle/kaggle-api#api-credentials.

    Args:
        dataset_name: the name of the dataset to download.
        download_root: path to the directory to download the dataset files to.
    """
    kaggle.api.competition_download_files(dataset_name, path=download_root)
