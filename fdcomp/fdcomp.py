from typing import List, Union
from pathlib import Path
from abc import abstractmethod
import numpy as np
import struct

from .encoder import Encoder, EncoderTRVL, EncoderTRVLVideo, EncoderRVLVideo
from .decoder import Decoder, DecoderTRVL, DecoderTRVLVideo, DecoderRVLVideo

DEFAULT_ENCODER = EncoderTRVLVideo
DEFAULT_DECODER = DecoderTRVL

def load(file: Union[str, Path], decoder: Union[Decoder, str] = None, *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from a file using the provided decoder.
    
    :param file: Path to the file containing the depth map.
    :param decoder: An instance of a Decoder subclass.
    :return: The decoded depth map as a numpy array.
    """
    if isinstance(decoder, str):
        if decoder.lower() == "trvl":
            decoder = DecoderTRVLVideo(*args, **kwargs)
        elif decoder.lower() == "rvl":
            decoder = DecoderRVLVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported decoder name: {decoder}")

    return Loader.load(file, decoder, *args, **kwargs)

def loads(data: Union[bytes, List[bytes]], decoder: Union[Decoder, str] = None, *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from raw bytes using the provided decoder.
    
    :param data: Raw bytes containing the depth map.
    :param decoder: An instance of a Decoder subclass.
    :return: The decoded depth map as a numpy array.
    """
    if isinstance(decoder, str):
        if decoder.lower() == "trvl":
            decoder = DecoderTRVLVideo(*args, **kwargs)
        elif decoder.lower() == "rvl":
            decoder = DecoderRVLVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported decoder name: {decoder}")

    if isinstance(data, bytes):
        data = [data]

    return decoder.decode(data, *args, **kwargs)


def dump(data: np.ndarray, encoder: Union[Encoder, str] = None, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """
    if isinstance(encoder, str):
        if encoder.lower() == "trvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderTRVLVideo(frame_size=frame_size, *args, **kwargs)
        elif encoder.lower() == "rvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderRVLVideo(frame_size=frame_size, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder}")

    return encoder(data, *args, **kwargs)

def save(data: np.ndarray, file: Union[str, Path], encoder: Union[Encoder, str] = None, *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass.
    """
    if isinstance(encoder, str):
        if encoder.lower() == "trvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderTRVLVideo(frame_size=frame_size, *args, **kwargs)
        elif encoder.lower() == "rvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderRVLVideo(frame_size=frame_size, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder}")
        
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


        if encoder.name == "TRVL":
            data_encoded, keyframes = encoder.encode(data)
        else:
            data_encoded = encoder.encode(data)
            keyframes = []


        # Open in binary mode
        with open(save_path, 'wb') as f:
            # Write header as ASCII, terminated by newline
            header = f"!{shape}; {dtype}; {encoder.name}; {keyframes}\n"
            f.write(header.encode('ascii'))
            for i in range(len(data_encoded)):
                block = data_encoded[i]  # raw bytes
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
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".dep")
        
        with open(path, 'rb') as f:
            # Read header line
            header = f.readline().decode('ascii').strip()[1:]
            header_parts = header.split('; ')
            data_add = None
            if len(header_parts) == 3:
                shape_str, dtype_str, enc_name = header_parts
            elif len(header_parts) == 4:
                shape_str, dtype_str, enc_name, data_add = header_parts
            shape = tuple(map(int, shape_str.strip("()").split(",")))
            dtype = np.dtype(dtype_str)

            frame_size = shape[-2] * shape[-1]

            # Instantiate decoder if needed
            if decoder is None:
                if enc_name == "TRVL":
                    decoder = DecoderTRVLVideo(shape[-2]*shape[-1], *args, **kwargs)
                elif enc_name == "RVL":
                    decoder = DecoderRVLVideo(shape[-2]*shape[-1], *args, **kwargs)
                else:
                    raise ValueError(f"Unsupported encoder name: {enc_name}")
            elif decoder.name != enc_name:
                raise ValueError(f"Decoder {decoder.name} does not match encoded data ({enc_name}).")

            frames = []
            frames_enc = []
            # Now read until EOF
            while True:
                # Read 4-byte length prefix
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # EOF
                (block_len,) = struct.unpack('>I', length_bytes)
                block = f.read(block_len)
                frames_enc.append(block)

        if enc_name == "TRVL":
            if data_add == None:
                keyframes = []
            else:
                keyframes = list(map(int, data_add.strip("[]").split(",")))

            arr = decoder(frames_enc, frame_size=frame_size, keyframes=keyframes)
        else:
            arr = decoder(frames_enc, frame_size=frame_size)

        arr = np.reshape(arr, shape)
        if 'float' in dtype_str:
            arr = arr.view(np.float16)

        arr = arr.astype(dtype)
        return arr