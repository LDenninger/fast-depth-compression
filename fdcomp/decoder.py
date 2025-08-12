import numpy as np
from typing import List, Union, Tuple
import io
from termcolor import colored
import time
from collections.abc import Sequence

try:
    from . import fdc_bindings as fb
except ImportError:
    raise ImportError("fdcomp C++ backend is not available. Make sure to have it build and installed correctly.")
# Export names
__all__ = [
    'Decoder', 'FrameDecoder', 'VideoDecoder',
    'DecoderRaw', 'DecoderRawVideo',
    'DecoderTRVL', 'DecoderTRVLVideo',
    'DecoderRVL', 'DecoderRVLVideo'
]

#######################################################
################### Base Classes ######################
#######################################################
class Decoder:
    name: str = "DecoderBase"
    def __init__(self, suppress_warnings: bool = True, decode_fn=None):
        self.suppress_warnings = suppress_warnings
        self._decode_fn = decode_fn

    def decode(
        self,
        data: Union[bytes, List[bytes]],
        output_size: Tuple[int, int] = None,
        dtype = np.int16,
        verbose: bool = False,
        *args, **kwargs
    ) -> np.ndarray:
        if self._decode_fn is None:
            raise NotImplementedError(
                "Decoder must be supplied with the decoding function "
                "('decode_fn' argument in constructor) or reimplement decode()."
            )
        if verbose:
            return self._decode_verbose(data, output_size, *args, **kwargs)

        data_uncompressed = self._decode_fn(data, *args, **kwargs)

        # If the binding already returns a NumPy array[int16], avoid copying
        if isinstance(data_uncompressed, np.ndarray):
            arr = data_uncompressed
            if arr.dtype != np.int16:
                arr = arr.astype(np.int16, copy=False)
        else:
            arr = np.asarray(data_uncompressed, dtype=np.int16)

        if dtype == np.float16:
            arr = arr.view(np.float16)
        elif dtype == np.float32:
            arr = arr.view(np.float16).astype(np.float32)
        else:
            arr = arr.astype(dtype)

        if output_size is not None:
            if arr.ndim == 1:
                arr = arr.reshape(*output_size)
            elif arr.ndim == 2 and np.prod(output_size) == arr.shape[1]:
                # already (frames, elems_per_frame); leave as-is or reshape to (frames, H, W)
                arr = arr.reshape(-1, *output_size)
            else:
                arr = arr.reshape(-1, *output_size)

        # arr is int16; NaN check is unnecessary (integers cannot be NaN).
        return arr

    def _decode_verbose(
        self,
        data: Union[bytes, List[bytes]],
        output_size: Tuple[int, int] = None,
        *args, **kwargs
    ) -> np.ndarray:
        if self._decode_fn is None:
            raise NotImplementedError(
                "Decoder must be supplied with the decoding function "
                "('decode_fn' argument in constructor) or reimplement decode()."
            )

        start_time = time.perf_counter()
        data_uncompressed = self._decode_fn(data, *args, **kwargs)
        end_time = time.perf_counter()
        print(f"Decoding time: {(end_time - start_time) * 1000:.2f} ms")

        start_time = time.perf_counter()
        if isinstance(data_uncompressed, np.ndarray):
            arr = data_uncompressed
            if arr.dtype != np.int16:
                arr = arr.astype(np.int16, copy=False)
        else:
            arr = np.asarray(data_uncompressed, dtype=np.int16)
        end_time = time.perf_counter()
        print(f"Conversion time: {(end_time - start_time) * 1000:.2f} ms")

        if output_size is not None:
            if arr.ndim == 1:
                arr = arr.reshape(*output_size)
            elif arr.ndim == 2 and np.prod(output_size, dtype=np.int64) == arr.shape[1]:
                arr = arr.reshape(-1, *output_size)
            else:
                arr = arr.reshape(-1, *output_size)

        return arr

    def _cast_int16(self, data: np.ndarray, suppress_warnings: bool = True) -> np.ndarray:
        if data.dtype == np.float16:
            if not suppress_warnings:
                print("Warning: Automatically converting float16 to int16 for encoding.")
            return data.view(np.int16)
        elif data.dtype == np.float32:
            if not suppress_warnings:
                print("Warning: Automatically narrowing float32 to int16 for encoding.")
            return data.astype(np.float16).view(np.int16)
        elif data.dtype == np.int32:
            if not suppress_warnings:
                print("Warning: Automatically narrowing int32 to int16 for encoding.")
            return data.astype(np.int16)
        return data


class FrameDecoder(Decoder, fb.FrameDecoder):
    """
    Base frame decoder binding wrapper.
    """
    name: str = "FrameDecoderBase"
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.FrameDecoder.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.FrameDecoder.decode.__get__(self, fb.FrameDecoder))

class VideoDecoder(Decoder, fb.VideoDecoder):
    """
    Base video decoder binding wrapper.
    """
    name: str = "VideoDecoderBase"
    def __init__(self, frame_decoder: FrameDecoder, suppress_warnings: bool = True):
        fb.VideoDecoder.__init__(self, frame_decoder)
        Decoder.__init__(self, suppress_warnings, fb.VideoDecoder.decode.__get__(self, fb.VideoDecoder))

#######################################################
#################### Raw Decoder ######################
#######################################################

class DecoderRaw(Decoder):
    """
    Raw frame decoder: loads numpy array from bytes.
    """
    name = "raw"
    def __init__(self, suppress_warnings: bool = True):
        Decoder.__init__(self, suppress_warnings)
    def decode(self, data: bytes) -> np.ndarray:
        buf = io.BytesIO(data)
        arr = np.load(buf)
        buf.close()
        return arr

class DecoderRawVideo(Decoder):
    """
    Raw video decoder: sequence of raw frame bytes.
    """
    name = "raw"
    def __init__(self, suppress_warnings: bool = True):
        Decoder.__init__(self, suppress_warnings)
        self.frame_decoder = DecoderRaw(suppress_warnings)
    def decode(self, data: List[bytes], *args, **kwargs) -> np.ndarray:
        frames = [self.frame_decoder.decode(b) for b in data]
        return np.stack(frames, axis=0)

#######################################################
#################### TRVL Decoder #####################
#######################################################
class DecoderTRVL(Decoder, fb.DecoderTRVL):
    """
    TRVL frame decoder wrapping the C++ binding.
    """
    name: str = "TRVL"
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.DecoderTRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.DecoderTRVL.decode.__get__(self, fb.DecoderTRVL))

class DecoderTRVLVideo(Decoder, fb.VideoDecoderTRVL):
    """
    TRVL video decoder wrapping the C++ binding.
    """
    name: str = "TRVL"
    def __init__(self,
                 frame_size: int = 0,
                 keyframe_interval: int = 10,
                 suppress_warnings: bool = True):
        # Validate keyframe_interval to prevent division by zero
        if keyframe_interval <= 0:
            keyframe_interval = 1  # Default to 1 if invalid
            if not suppress_warnings:
                print(colored("Warning: ", "yellow"), f"Invalid keyframe_interval ({keyframe_interval}), setting to 1")
        
        # Initialize C++ binding with correct parameter order: keyframe_interval, frame_size
        fb.VideoDecoderTRVL.__init__(self, keyframe_interval, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.VideoDecoderTRVL.decode.__get__(self, fb.VideoDecoderTRVL))

#######################################################
#################### RVL Decoder ######################
#######################################################
class DecoderRVL(Decoder, fb.DecoderRVL):
    """
    RVL frame decoder wrapping the C++ binding.
    """
    name: str = "RVL"
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.DecoderRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.DecoderRVL.decode.__get__(self, fb.DecoderRVL))

class DecoderRVLVideo(Decoder, fb.VideoDecoderRVL):
    """
    RVL video decoder wrapping the optimized C++ binding that returns a NumPy array[int16].
    """
    name: str = "RVL"
    def __init__(self, frame_size: int = 0, suppress_warnings: bool = True):
        fb.VideoDecoderRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.VideoDecoderRVL.decode.__get__(self, fb.VideoDecoderRVL))

