import io
import numpy as np
from typing import List, Union
from termcolor import colored
import time

try:
    from . import fdc_bindings as fb
    #import fdc_bindings as fb
except ImportError:
    raise ImportError("fdcomp C++ backend is not available. Make sure to have it build and installed correctly.")


# Export names
__all__ = [
    'Encoder', 'FrameEncoder', 'VideoEncoder',
    'EncoderRaw', 'EncoderRawVideo',
    'EncoderTRVL', 'EncoderTRVLVideo',
    'EncoderRVL', 'EncoderRVLVideo'
]

#######################################################
################### Base Classes ######################
#######################################################
class Encoder:
    def __init__(self, 
                 suppress_warnings: bool = True,
                 encode_fn = None,
                 ):
        self.suppress_warnings = suppress_warnings
        self._encode_fn = encode_fn
        return
    
    def encode(self, data: np.ndarray, verbose: bool = False, *args, **kwargs) -> Union[List[bytes],bytes]:
        if verbose:
            return self._encode_verbose(data, *args, **kwargs)
        if self._encode_fn is None:
            raise NotImplementedError("Encoder must be supplied with the encoding function ('encode_fn' argument in constructor) or reimplement the encode() function.")
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).ravel().tolist()
        data_compressed = self._encode_fn(data, *args, **kwargs)
        return data_compressed
    
    def _encode_verbose(self, data: np.ndarray, *args, **kwargs) -> Union[List[bytes],bytes]:
        if self._encode_fn is None:
            raise NotImplementedError("Encoder must be supplied with the encoding function ('encode_fn' argument in constructor) or reimplement the encode() function.")
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        start_time = time.perf_counter()
        data = np.ascontiguousarray(data).ravel().tolist()
        end_time = time.perf_counter()
        conversion_time = (end_time - start_time) * 1000  # in milliseconds
        print(f"Conversion time: {conversion_time:.2f} ms")
        start_time = time.perf_counter()
        data_compressed = self._encode_fn(data, *args, **kwargs)
        end_time = time.perf_counter()
        encoding_time = (end_time - start_time) * 1000  # in milliseconds
        print(f"Encoding time: {encoding_time:.2f} ms")
        return data_compressed

    
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

class FrameEncoder(Encoder, fb.FrameEncoder):
    """
    Base frame encoder binding wrapper.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        # call C++ binding init first
        fb.FrameEncoder.__init__(self, frame_size)
        # then initialize our Encoder base
        Encoder.__init__(self, suppress_warnings, fb.FrameEncoder.encode.__get__(self, fb.FrameEncoder))

class VideoEncoder(Encoder, fb.VideoEncoder):
    """
    Base video encoder binding wrapper.
    """
    def __init__(self, frame_encoder: FrameEncoder, suppress_warnings: bool = True):
        fb.VideoEncoder.__init__(self, frame_encoder)
        Encoder.__init__(self, suppress_warnings, fb.VideoEncoder.encode.__get__(self, fb.VideoEncoder))

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        if self._encode_fn is None:
            raise NotImplementedError("Encoder must be supplied with the encoding function ('encode_fn' argument in constructor) or reimplement the encode() function.")
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).ravel().tolist()
        data_compressed = self._encode_fn(data, *args, **kwargs)
        return data_compressed

#######################################################
################### Raw Encoder #######################
#######################################################
class EncoderRaw(Encoder):
    """
    Raw frame encoder: saves numpy array as bytes.
    """
    name = "raw"
    def __init__(self, suppress_warnings: bool = True):
        Encoder.__init__(self, suppress_warnings)

    def encode(self, data: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, data)
        dbytes = buf.getvalue()
        buf.close()
        return dbytes

class EncoderRawVideo(Encoder):
    """
    Raw video encoder: sequence of raw frame bytes.
    """
    name = "raw"
    def __init__(self, suppress_warnings: bool = True):
        Encoder.__init__(self, suppress_warnings)
        # pass warning flag into the raw frame encoder
        self.frame_encoder = EncoderRaw(suppress_warnings)

    def encode(self, data: np.ndarray) -> List[bytes]:
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return [self.frame_encoder.encode(frame) for frame in data]

#######################################################
################### TRVL Encoder ######################
#######################################################
class EncoderTRVL(Encoder, fb.EncoderTRVL):
    """
    TRVL frame encoder wrapping the C++ binding.
    """
    def __init__(self,
                 frame_size: int,
                 change_threshold: int = 10,
                 invalidation_threshold: int = 2,
                 suppress_warnings: bool = True):
        fb.EncoderTRVL.__init__(self, frame_size, change_threshold, invalidation_threshold)
        Encoder.__init__(self, suppress_warnings, fb.EncoderTRVL.encode.__get__(self, fb.EncoderTRVL))

class EncoderTRVLVideo(Encoder, fb.VideoEncoderTRVL):
    """
    TRVL video encoder wrapping the C++ binding.
    """
    def __init__(self,
                 frame_size: int,
                 keyframe_interval: int = 10,
                 change_threshold: int = 10,
                 invalidation_threshold: int = 2,
                 suppress_warnings: bool = True):
        # Validate keyframe_interval to prevent division by zero
        if keyframe_interval <= 0:
            keyframe_interval = 1  # Default to 1 if invalid
            if not suppress_warnings:
                print(colored("Warning: ", "yellow"), f"Invalid keyframe_interval ({keyframe_interval}), setting to 1")
        
        # Initialize C++ binding with correct parameter order: keyframe_interval, frame_size, change_threshold, invalidation_threshold
        fb.VideoEncoderTRVL.__init__(self,
                                     keyframe_interval,
                                     frame_size,
                                     change_threshold,
                                     invalidation_threshold)
        Encoder.__init__(self, suppress_warnings)

    def encode(self, data: np.ndarray, *args, **kwargs) -> List[bytes]:
        # Cast data to int16 and make contiguous
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        
        # Calculate number of frames from the original shape before flattening
        num_frames = data.shape[0]
        
        # Now flatten for the C++ binding
        data = np.ascontiguousarray(data).ravel().tolist()
        
        # Call the C++ binding directly with positional arguments
        data_compressed = fb.VideoEncoderTRVL.encode(self, data, num_frames)
        return data_compressed
#######################################################
#################### RVL Encoder ######################
#######################################################
class EncoderRVL(Encoder, fb.EncoderRVL):
    """
    RVL frame encoder wrapping the C++ binding.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        fb.EncoderRVL.__init__(self, frame_size)
        Encoder.__init__(self, suppress_warnings, fb.EncoderRVL.encode.__get__(self, fb.EncoderRVL))

class EncoderRVLVideo(Encoder, fb.VideoEncoderRVL):
    """
    RVL video encoder wrapping the C++ binding.
    """
    def __init__(self, frame_size: int, suppress_warnings: bool = True):
        self.frame_size = frame_size
        fb.VideoEncoderRVL.__init__(self, frame_size)
        Encoder.__init__(self, suppress_warnings, fb.VideoEncoderRVL.encode.__get__(self, fb.VideoEncoderRVL))

    def encode(self, data: np.ndarray, *args, **kwargs) -> List[bytes]:
        num_frames = data.shape[0]
        return super().encode(data, *args, num_frames=num_frames, **kwargs)

