from typing import List, Union
from pathlib import Path
from abc import abstractmethod
import numpy as np
import struct

from .encoder import Encoder, EncoderTRVL
from .decoder import Decoder, DecoderTRVL

DEFAULT_ENCODER = EncoderTRVL
DEFAULT_DECODER = DecoderTRVL

def load(file: Union[str, Path], decoder: Decoder = None, *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from a file using the provided decoder.
    
    :param file: Path to the file containing the depth map.
    :param decoder: An instance of a Decoder subclass.
    :return: The decoded depth map as a numpy array.
    """

    return Loader.load(file, decoder, *args, **kwargs)


def dump(data: np.ndarray, encoder: Encoder = None, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """

    return encoder(data, *args, **kwargs)

def save(data: np.ndarray, file: Union[str, Path], encoder: Encoder = None, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """
    Saver.save(data, file, encoder, *args, **kwargs)


class Saver:

    @classmethod
    def save(cls, data: np.ndarray, save_path: str, encoder: Encoder = None, *args, **kwargs):
        shape = data.shape
        dtype = data.dtype
        encoder = encoder or DEFAULT_ENCODER(frame_size=shape[-1] * shape[-2], *args, **kwargs)

        if data.ndim <= 2:
            data = data[np.newaxis]
        elif data.ndim > 3:
            data = np.reshape(data, (-1, shape[-2], shape[-1]))

        if Path(save_path).suffix == "":
            save_path += ".dep"

        encoder = encoder or DEFAULT_ENCODER(frame_size=shape[-1] * shape[-2], *args, **kwargs)

        # Open in binary mode
        with open(save_path, 'wb') as f:
            # Write header as ASCII, terminated by newline
            header = f"!{shape}; {dtype}; {encoder.name}\n"
            f.write(header.encode('ascii'))

            for i in range(data.shape[0]):
                block = encoder(data[i], *args, **kwargs)  # raw bytes
                # First write a 4-byte big-endian length:
                f.write(struct.pack('>I', len(block)))
                # Then the raw compressed bytes
                f.write(block)

class Loader:

    @classmethod
    def load(cls, path: str, decoder: Decoder = None, *args, **kwargs) -> List[np.ndarray]:
        """
        Load a depth map from a file using the provided decoder.
        
        :param path: Path to the file containing the depth map.
        :param decoder: An instance of a Decoder subclass.
        :return: A list of decoded depth maps as numpy arrays.
        """
        with open(path, 'rb') as f:
            # Read header line
            header = f.readline().decode('ascii').strip()[1:]
            shape_str, dtype_str, enc_name = header.split('; ')
            shape = tuple(map(int, shape_str.strip("()").split(",")))
            dtype = np.dtype(dtype_str)

            # Instantiate decoder if needed
            if decoder is None:
                if enc_name == "TRVL":
                    decoder = DecoderTRVL(shape[-2]*shape[-1], *args, **kwargs)
                elif enc_name == "RVL":
                    decoder = Decoder(shape[-2]*shape[-1], *args, **kwargs)
                else:
                    raise ValueError(f"Unsupported encoder name: {enc_name}")
            elif decoder.name != enc_name:
                raise ValueError(f"Decoder {decoder.name} does not match encoded data ({enc_name}).")

            frames = []
            # Now read until EOF
            while True:
                # Read 4-byte length prefix
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # EOF
                (block_len,) = struct.unpack('>I', length_bytes)
                block = f.read(block_len)
                frames.append(decoder(block, *args, **kwargs))

        arr = np.stack(frames, axis=0)
        arr = np.reshape(arr, shape)
        if 'float' in dtype_str:
            arr = arr.view(np.float16)

        arr = arr.astype(dtype)
        return arr