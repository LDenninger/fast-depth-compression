from typing import List, Literal, Union
from pathlib import Path
from abc import abstractmethod
import numpy as np
import struct

from .encoder import Encoder, EncoderTRVL, EncoderTRVLVideo, EncoderRVLVideo
from .decoder import Decoder, DecoderTRVL, DecoderTRVLVideo, DecoderRVLVideo

DEFAULT_ENCODER = EncoderTRVLVideo
DEFAULT_DECODER = DecoderTRVL

class FileMeta:
    """
    Class to hold metadata about a file, such as shape, dtype, and encoder type.
    """
    def __init__(self, path: str, shape: tuple, dtype: str, encoder: str, data_add: str = None):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.encoder = encoder
        self.data_add = data_add

    def __repr__(self):
        return f"fdcompFile(shape={self.shape}, dtype={self.dtype}, encoder={self.encoder}, data_add={self.data_add})"

def inspect(file: Union[str, Path],
            return_result: bool = True,
            print_result: bool = True,
            printf: callable = print) -> Union[dict, None]:
    
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    
    if file.suffix in ['.npy', '.npz', '.np']:
        np_arr = np.load(file, allow_pickle=True)
        enc_name = "numpy"
        shape = np_arr.shape
        dtype = np_arr.dtype
        data_add = None
    else:
    
        try:
            with open(file, 'rb') as f:
                # Read header line
                header = f.readline().decode('ascii').strip()[1:]
                header_parts = header.split('; ')
                
                if len(header_parts) == 3:
                    shape_str, dtype_str, enc_name = header_parts
                    data_add = None
                elif len(header_parts) == 4:
                    shape_str, dtype_str, enc_name, data_add = header_parts
                else:
                    raise ValueError(f"Invalid header format in {file}")

                shape = tuple(map(int, shape_str.strip("()").split(",")))
                dtype = np.dtype(dtype_str)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return None

    result = {
        "file": file,
        "suffix": Path(file).suffix,
        "file_size": Path(file).stat().st_size if Path(file).exists() else None,
        "compression_type": enc_name,
        "resolution": shape[1:],
        "num_frames": shape[0],
        "dtype": dtype,
        "Additional Data": data_add,
    }

    if print_result:
        printf(f"{result['file']}")
        printf(f" suffix:............ {result['suffix']}")
        printf(f" file Size:......... {result['file_size']} bytes")
        printf(f" compression Type:.. {result['compression_type']}")
        printf(f" resolution:........ {result['resolution']}")
        printf(f" number of Frames:.. {result['num_frames']}")
        printf(f" dtype:............. {result['dtype']}")
        if data_add is not None:
            printf(f" additional Data:... {data_add}")
    if return_result:
        return result
    return 

def load(file: Union[str, Path],
         decoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None,
         return_meta: bool = False,
         *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from a file using the provided decoder.
    
    :param file: Path to the file containing the depth map.
    :param decoder: An instance of a Decoder subclass or a string indicating the decoder type. Available options: ['trvl', 'rvl', 'raw'].
    :return: The decoded depth map as a numpy array.
    """
    if isinstance(decoder, str):
        if decoder.lower() == "trvl":
            decoder = DecoderTRVLVideo(*args, **kwargs)
        elif decoder.lower() == "rvl":
            decoder = DecoderRVLVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported decoder name: {decoder}")

    return Loader.load(file, decoder, return_meta, *args, **kwargs)

def loads(data: Union[bytes, List[bytes]], 
          decoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None, 
          *args, **kwargs) -> np.ndarray:
    """
    Load a depth map from raw bytes using the provided decoder.
    
    :param data: Raw bytes containing the depth map.
    :param decoder: An instance of a Decoder subclass or a string indicating the decoder type. Available options: ['trvl', 'rvl', 'raw'].
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


def dump(data: np.ndarray, 
         encoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None, 
         *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.
    :param encoder: An instance of an Encoder subclass or a string indicating the encoder type. Available options: ['trvl', 'rvl', 'raw'].
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

def save(data: np.ndarray, 
         file: Union[str, Path] = None, 
         encoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None, 
         meta: FileMeta = None,
         *args, **kwargs) -> None:
    """
    Save a depth map to a file using the provided encoder.
    
    :param data: The depth map to be saved as a numpy array.
    :param file: Path to the file where the depth map will be saved.  Optional if `meta` is provided.
    :param encoder: An instance of an Encoder subclass or a string indicating the encoder type. Available options: ['trvl', 'rvl', 'raw'].
    :param meta: Optional metadata about the file, such as shape, dtype, and encoder type.
    """
    if file is None:
        assert meta is not None, "Either 'file' or 'meta' must be provided."
        file = meta.path
    if encoder is None:
        if meta is not None:
            encoder = meta.encoder
        encoder = DEFAULT_ENCODER(frame_size=data.shape[-1] * data.shape[-2], *args, **kwargs)

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
        save_path = Path(save_path)

        if data.ndim <= 2:
            data = data[np.newaxis]
        elif data.ndim > 3:
            data = np.reshape(data, (-1, shape[-2], shape[-1]))

        if save_path.suffix == "":
            save_path = save_path.with_suffix(".dep")
        elif save_path.suffix != ".dep":
            save_path = save_path.with_suffix(".dep")


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
    def load(cls, path: str, decoder: Decoder = None, return_meta: bool = False, *args, **kwargs) -> List[np.ndarray]:
        """
        Load a depth map from a file using the provided decoder.
        
        :param path: Path to the file containing the depth map.
        :param decoder: An instance of a Decoder subclass.
        :return: A list of decoded depth maps as numpy arrays.
        """
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".dep")
        elif path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            meta = FileMeta(path, arr.shape, str(arr.dtype), "raw")
            return arr, meta if return_meta else arr
        
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
        return arr, FileMeta(path, shape, dtype_str, enc_name, data_add) if return_meta else arr