import numpy as np
from pathlib import Path

from .logger import get_logger

FILEMAP = {
    50: "jica_datarequest_nr0.npz",
    79: "jica_datarequest_nr1.npz",
}


def find_file_from_id(id: None | int = None) -> Path | None:
    """Locate file with data from id

    :param id: id of dataset according to mission data products
    :return: path of the file
    """
    logger = get_logger()
    fm_keys = FILEMAP.keys()
    ddir = get_datadir()

    if id in fm_keys:
        file = FILEMAP[id]
        filepath = ddir / file
        if filepath.exists():
            return filepath
    else:
        logger.error(f"File with id={id} could not be located in {ddir}")

    return None


def find_file_from_name(name: None | str = None) -> Path | None:
    """Locate file with data from filename

    :param name: name of the file
    :return:  path of the file
    """
    logger = get_logger()
    ddir = get_datadir()
    if name is not None:
        filepath = ddir / name
        if filepath.exists():
            return filepath
        else:
            logger.error(f"File {name} can not be located in {ddir}")
    return None


def get_datadir() -> Path:
    """Locate data directory

    :return: path to the data directory
    """
    fdir = Path(__file__)
    target = "data"

    for parent in fdir.parents:
        current = parent / target
        if current.exists():
            return current
