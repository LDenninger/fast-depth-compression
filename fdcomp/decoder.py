import numpy as np
from typing import List, Union, Tuple
import io
from termcolor import colored
import time

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

    def __init__(self, suppress_warnings: bool = True, decode_fn = None):
        self.suppress_warnings = suppress_warnings
        self._decode_fn = decode_fn
        return
    
    def decode(self, data: Union[bytes, List[bytes]], frame_size: Tuple[int, int] = None, verbose: bool = False, *args, **kwargs) -> np.ndarray:
        
        if verbose:
            return self._decode_verbose(data, frame_size, *args, **kwargs)
        
        if self._decode_fn is None:
            raise NotImplementedError("Decoder must be supplied with the decoding function ('decode_fn' argument in constructor) or reimplement the decode() function.")

        data_uncompressed = self._decode_fn(data, *args, **kwargs)
        data_uncompressed = np.array(data_uncompressed, dtype=np.int16)
        if frame_size is not None:
            if data_uncompressed.ndim == 1:
                data_uncompressed = data_uncompressed.reshape(*frame_size)
            else:
                data_uncompressed = data_uncompressed.reshape(-1, *frame_size)
            
        invalid_data = np.isnan(data_uncompressed)
        data_uncompressed[invalid_data] = 0 
        return data_uncompressed
    
    def _decode_verbose(self, data: Union[bytes, List[bytes]], frame_size: Tuple[int, int] = None, *args, **kwargs) -> np.ndarray:
        if self._decode_fn is None:
            raise NotImplementedError("Decoder must be supplied with the decoding function ('decode_fn' argument in constructor) or reimplement the decode() function.")

        start_time = time.perf_counter()
        data_uncompressed = self._decode_fn(data, *args, **kwargs)
        end_time = time.perf_counter()
        decoding_time = (end_time - start_time) * 1000  # in milliseconds
        print(f"Decoding time: {decoding_time:.2f} ms")
        start_time = time.perf_counter()
        data_uncompressed = np.array(data_uncompressed, dtype=np.int16)
        end_time = time.perf_counter()
        conversion_time = (end_time - start_time) * 1000  # in milliseconds
        print(f"Conversion time: {conversion_time:.2f} ms")
        if frame_size is not None:
            if data_uncompressed.ndim == 1:
                data_uncompressed = data_uncompressed.reshape(*frame_size)
            else:
                data_uncompressed = data_uncompressed.reshape(-1, *frame_size)
            
        invalid_data = np.isnan(data_uncompressed)
        data_uncompressed[invalid_data] = 0 
        return data_uncompressed
    
    def _cast_int16(self, data: np.ndarray, suppress_warnings: bool = True) -> np.ndarray:
        """
            Casts the input data to int16 if it is of type float16.
        """
        if data.dtype == np.float16:
            if not suppress_warnings: print(colored("Warning: ", "yellow"), "Automatically converting float16 to int16 for encoding.")
            return data.view(np.int16)
        elif data.dtype == np.float32:
            if not suppress_warnings: print(colored("Warning: ", "yellow"), "Automatically narrowing float32 to int16 for encoding.")
            return data.astype(np.float16).view(np.int16)
        elif data.dtype == np.int32:
            if not suppress_warnings: print(colored("Warning: ", "yellow"), "Automatically narrowing int32 to int16 for encoding.")
            return data.astype(np.int16)
        return data

class FrameDecoder(Decoder, fb.FrameDecoder):
    """
    Base frame decoder binding wrapper.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.FrameDecoder.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.FrameDecoder.decode.__get__(self, fb.FrameDecoder))

class VideoDecoder(Decoder, fb.VideoDecoder):
    """
    Base video decoder binding wrapper.
    """
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
    def decode(self, data: List[bytes]) -> np.ndarray:
        frames = [self.frame_decoder.decode(b) for b in data]
        return np.stack(frames, axis=0)

#######################################################
#################### TRVL Decoder #####################
#######################################################
class DecoderTRVL(Decoder, fb.DecoderTRVL):
    """
    TRVL frame decoder wrapping the C++ binding.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.DecoderTRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.DecoderTRVL.decode.__get__(self, fb.DecoderTRVL))

class DecoderTRVLVideo(Decoder, fb.VideoDecoderTRVL):
    """
    TRVL video decoder wrapping the C++ binding.
    """
    def __init__(self,
                 frame_size: int,
                 keyframe_interval: int = 20,
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
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.DecoderRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.DecoderRVL.decode.__get__(self, fb.DecoderRVL))

class DecoderRVLVideo(Decoder, fb.VideoDecoderRVL):
    """
    RVL video decoder wrapping the C++ binding.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.VideoDecoderRVL.__init__(self, frame_size)
        Decoder.__init__(self, suppress_warnings, fb.VideoDecoderRVL.decode.__get__(self, fb.VideoDecoderRVL))



