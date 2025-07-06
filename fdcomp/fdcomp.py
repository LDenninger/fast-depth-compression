from typing import List, Union
from pathlib import Path
from abc import abstractmethod
import numpy as np

from .encoder import Encoder
from .decoder import Decoder


def load(file: Union[str, Path], decoder: Decoder, *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from a file using the provided decoder.
    
    :param file: Path to the file containing the depth map.
    :param decoder: An instance of a Decoder subclass.
    :return: The decoded depth map as a numpy array.
    """
    with open(file, 'rb') as f:
        data = f.read()
    return decoder(data, *args, **kwargs)

def dump(data: np.ndarray, encoder: Encoder, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """
    return encoder(data, *args, **kwargs)

def save(data: np.ndarray, file: Union[str, Path], encoder: Encoder, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """
    with open(file, 'wb') as f:
        f.write(encoder(data, *args, **kwargs))
