from typing import List, Literal, Union, Tuple
from pathlib import Path
from abc import abstractmethod
import numpy as np
import struct

from .encoder import Encoder, FrameEncoder, VideoEncoder, EncoderTRVL, EncoderTRVLVideo, EncoderRVLVideo, EncoderRawVideo
from .decoder import Decoder, FrameDecoder, VideoDecoder, DecoderTRVL, DecoderTRVLVideo, DecoderRVLVideo, DecoderRawVideo

DEFAULT_FRAME_ENCODER = EncoderTRVL
DEFAULT_VIDEO_ENCODER = EncoderTRVLVideo

DEFAULT_FRAME_DECODER = DecoderTRVL
DEFAULT_VIDEO_DECODER = DecoderTRVLVideo

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
    """
    Inspect a compressed depth file and extract metadata information.
    
    Supports both custom compressed formats (.dep) and numpy formats (.npy, .npz, .np).
    Reads file headers to extract shape, dtype, encoder type, and additional data.
    
    :param file: Path to the file to inspect.
    :param return_result: Whether to return the metadata as a dictionary.
    :param print_result: Whether to print the metadata to console.
    :param printf: Function to use for printing (defaults to built-in print).
    :return: Dictionary containing file metadata if return_result is True, otherwise None.
    :raises FileNotFoundError: If the specified file does not exist.
    """

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
         decoder: Union[Literal['trvl', 'rvl', 'raw']] = None,
         return_meta: bool = False,
         *args, **kwargs) -> np.ndarray:
    """
    Load and decode a depth map from a compressed file.
    
    Automatically detects the compression format from file headers and uses the 
    appropriate decoder. Supports TRVL, RVL, and raw compression formats.
    
    :param file: Path to the file containing the compressed depth map.
    :param decoder: Decoder type to use. If None, auto-detects from file header.
                   Available options: ['trvl', 'rvl', 'raw'].
    :param return_meta: Whether to return metadata along with the decoded data.
    :param args: Additional arguments passed to the decoder.
    :param kwargs: Additional keyword arguments passed to the decoder.
    :return: The decoded depth map as a numpy array, optionally with metadata.
    :raises ValueError: If an unsupported decoder type is specified.
    """
    if isinstance(decoder, str):
        if decoder.lower() == "trvl":
            decoder = DecoderTRVLVideo(*args, **kwargs)
        elif decoder.lower() == "rvl":
            decoder = DecoderRVLVideo(*args, **kwargs)
        elif decoder.lower() == "raw":
            decoder = DecoderRawVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported decoder name: {decoder}")
    out = Loader.load(file, decoder, return_meta, *args, **kwargs)
    return out

def loads(data: Union[bytes, List[bytes]], 
          decoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None, 
          output_size: Tuple[int,int] = None, dtype = np.int16,
          *args, **kwargs) -> np.ndarray:
    """
    Load and decode depth maps from raw compressed bytes.
    
    Decodes depth data directly from memory without file I/O. Useful for 
    streaming applications or when working with compressed data in memory.
    
    :param data: Raw bytes or list of byte arrays containing compressed depth data.
    :param decoder: Decoder instance or string specifying decoder type.
                   Available options: ['trvl', 'rvl', 'raw'].
    :param output_size: Expected output dimensions (height, width) for the decoded frames.
    :param dtype: Data type for the decoded array (default: np.int16).
    :param args: Additional arguments passed to the decoder.
    :param kwargs: Additional keyword arguments passed to the decoder.
    :return: The decoded depth map(s) as a numpy array.
    :raises ValueError: If decoder type is unsupported or required parameters are missing.
    """
    if isinstance(decoder, str):
        if decoder.lower() == "trvl":
            if 'frame_size' not in kwargs:
                raise ValueError("When using TRVL or RVL encoder, it is required to pass the 'frame_size' arguments to loads()")
            decoder = DecoderTRVLVideo(*args, **kwargs)
        elif decoder.lower() == "rvl":
            if 'frame_size' not in kwargs:
                raise ValueError("When using RVL or TRVL encoder, it is required to pass the 'frame_size' arguments to loads()")
            decoder = DecoderRVLVideo(*args, **kwargs)
        elif decoder.lower() == "raw":
            decoder = DecoderRawVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported decoder name: {decoder}")

    if isinstance(data, bytes):
        data = [data]

    return decoder.decode(data, output_size=output_size, dtype=dtype)


def dump(data: np.ndarray, 
         encoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = None, 
         *args, **kwargs) -> None:
    """
    Encode depth map data to compressed bytes without saving to file.
    
    Compresses depth data using the specified encoder and returns the raw 
    compressed bytes. Useful for streaming or in-memory compression operations.
    
    :param data: The depth map(s) to be compressed as a numpy array.
    :param encoder: Encoder instance or string specifying encoder type.
                   Available options: ['trvl', 'rvl', 'raw'].
    :param args: Additional arguments passed to the encoder.
    :param kwargs: Additional keyword arguments passed to the encoder.
    :return: Compressed data as bytes.
    :raises ValueError: If an unsupported encoder type is specified.
    """

    if isinstance(encoder, str):
        if encoder.lower() == "trvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderTRVLVideo(frame_size=frame_size, *args, **kwargs)
        elif encoder.lower() == "rvl":
            frame_size = data.shape[-1] * data.shape[-2]
            encoder = EncoderRVLVideo(frame_size=frame_size, *args, **kwargs)
        elif encoder.lower() == "raw":
            encoder = EncoderRawVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder}")

    return encoder.encode(data, *args, **kwargs)

def save(data: np.ndarray, 
         file: Union[str, Path] = None, 
         encoder: Union[Decoder, Literal['trvl', 'rvl', 'raw']] = "trvl", 
         meta: FileMeta = None,
         *args, **kwargs) -> None:
    """
    Compress and save depth map data to a file.
    
    Encodes depth data using the specified compression algorithm and saves it
    with appropriate headers for later decoding. Automatically handles file
    extension and format detection.
    
    :param data: The depth map(s) to be saved as a numpy array.
    :param file: Path where the compressed depth map will be saved. 
                Optional if meta parameter contains path information.
    :param encoder: Encoder instance or string specifying compression type.
                   Available options: ['trvl', 'rvl', 'raw']. Defaults to 'trvl'.
    :param meta: Optional FileMeta object containing file metadata and path information.
    :param args: Additional arguments passed to the encoder.
    :param kwargs: Additional keyword arguments passed to the encoder.
    :raises ValueError: If encoder type is unsupported.
    :raises AssertionError: If neither file nor meta parameter is provided.
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
        elif encoder.lower() == "raw":
            encoder = EncoderRawVideo(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder}")
        
    Saver.save(data, file, encoder, *args, **kwargs)


class Saver:
    """
    Static class for saving compressed depth data to files.
    
    Handles the low-level file writing operations, including header generation,
    data encoding, and binary file format management.
    """

    @classmethod
    def save(cls, data: np.ndarray, save_path: str, encoder: Encoder = None, *args, **kwargs):
        """
        Save numpy array data to a compressed depth file format.
        
        Writes a custom binary format with ASCII header containing metadata
        followed by compressed frame data blocks with length prefixes.
        
        :param data: Numpy array containing depth data to compress and save.
        :param save_path: File path where the compressed data will be written.
        :param encoder: Encoder instance to use for compression. Uses default if None.
        :param args: Additional arguments passed to the encoder.
        :param kwargs: Additional keyword arguments passed to the encoder.
        """
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



        data_encoded = encoder.encode(data)
        keyframes = []

        with open(save_path, 'wb') as f:
            header = f"!{shape}; {dtype}; {encoder.name}; {keyframes}\n"
            f.write(header.encode('ascii'))
            for i in range(len(data_encoded)):
                block = data_encoded[i]  
                f.write(struct.pack('>I', len(block)))
                f.write(block)

class Loader:
    """
    Static class for loading compressed depth data from files.
    
    Handles the low-level file reading operations, including header parsing,
    data decoding, and binary file format management.
    """

    @classmethod
    def load(cls, path: str, decoder: Decoder = None, return_meta: bool = False, *args, **kwargs) -> List[np.ndarray]:
        """
        Load compressed depth data from file and decode to numpy arrays.
        
        Reads custom binary format files with ASCII headers, automatically
        detects compression type, and decodes the data using appropriate decoder.
        Also supports loading standard numpy (.npy) files.
        
        :param path: Path to the compressed depth file to load.
        :param decoder: Decoder instance to use. Auto-detects from file if None.
        :param return_meta: Whether to return FileMeta object along with data.
        :param args: Additional arguments passed to the decoder.
        :param kwargs: Additional keyword arguments passed to the decoder.
        :return: Decoded depth data as numpy array, optionally with metadata.
        :raises ValueError: If decoder doesn't match file format or unsupported format.
        """
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".dep")
        elif path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            meta = FileMeta(path, arr.shape, str(arr.dtype), "raw")
            return arr, meta if return_meta else arr
        
        with open(path, 'rb') as f:
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

            if decoder is None:
                if enc_name == "TRVL":
                    decoder = DecoderTRVLVideo(shape[-2]*shape[-1], *args, **kwargs)
                elif enc_name == "RVL":
                    decoder = DecoderRVLVideo(shape[-2]*shape[-1], *args, **kwargs)
                elif enc_name == "raw":
                    decoder = DecoderRawVideo(*args, **kwargs)
                else:
                    raise ValueError(f"Unsupported encoder name: {enc_name}")
            elif decoder.name != enc_name:
                raise ValueError(f"Decoder {decoder.name} does not match encoded data ({enc_name}).")

            frames = []
            frames_enc = []
            while True:
                length_bytes = f.read(4)
                if not length_bytes:
                    break 
                (block_len,) = struct.unpack('>I', length_bytes)
                block = f.read(block_len)
                frames_enc.append(block)

        if enc_name == "TRVL":
            # TODO: Fix this for providing keyframes
            #if data_add == None or data_add.strip("[]") == "":
            keyframes = []
            arr = decoder.decode(frames_enc, frame_size=frame_size)
            #else:
            #    keyframes = list(map(int, data_add.strip("[]").split(",")))
            #    arr = decoder.decode(frames_enc, frame_size=frame_size, keyframes=keyframes)
        else:
            arr = decoder.decode(frames_enc, frame_size=frame_size)

        arr = np.reshape(arr, shape)
        if 'float' in dtype_str:
            arr = arr.view(np.float16)

        arr = arr.astype(dtype)
        if return_meta: return arr, FileMeta(path, shape, dtype_str, enc_name, data_add)
        else: return arr