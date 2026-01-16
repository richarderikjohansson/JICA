import numpy as np
from pathlib import Path
from types import SimpleNamespace

from .logger import get_logger

FILEMAP = {
    50: "jica_datarequest_nr0.npz",
    79: "jica_datarequest_nr2_and_nr1.npz",
    68: "jica_datarequest_nr4.npz",
    65: "jica_datarequest_nr6.npz",
    80: "jica_datarequest_nr7.npz",
    61: "jica_datarequest_nr8.npz",
    73: "jica_datarequest_nr9.npz",
    81: "jica_datarequest_nr10.npz",
    69: "jica_datarequest_nr11.npz",
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


def read_data_from_id(id: int) -> SimpleNamespace:
    """Function to read data from id

    :param id: id of dataset according to mission data products
    :return: simple name space of the data
    """
    fp = find_file_from_id(id=id)
    npdata = np.load(fp)
    dct = {k: npdata[k] for k in npdata.keys()}
    namespace = SimpleNamespace(**dct)
    return namespace


def read_data_from_name(name: str) -> SimpleNamespace:
    """Function to read data from filename

    :param name: name of the file located in /data
    :return: simple name space of the data
    """
    fp = find_file_from_name(name)
    npdata = np.load(fp)
    dct = {k: npdata[k] for k in npdata.keys()}
    namespace = SimpleNamespace(**dct)
    return namespace
