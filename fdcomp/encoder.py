import io

from typing import List

import numpy as np
from termcolor import colored

# Import bindings with fallback
def _get_bindings():
    try:
        from .fdc_bindings import EncoderTRVL as _EncoderTRVL, RVLCompress
        return _EncoderTRVL, RVLCompress
    except ImportError:
        # If importing during development without compiled bindings
        return None, None

_EncoderTRVL, RVLCompress = _get_bindings()

############################################
############### Base Classes ###############
############################################

class Encoder:
    """
        Base class for all encoders.
        Implemented encoders should inherit from either the subclasses FrameEncoder or VideoEncoder.
    
    """
    def __call__(self, data: np.ndarray, *args, **kwargs) -> bytes:
        """ 
            Calls the encoding function.
        """
        return self.encode(data, *args, **kwargs)

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        """
            The encoding function to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement the encode() method.")

    def _cast_int16(self, data: np.ndarray, suppress_warnings: bool = False) -> np.ndarray:
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
    
class FrameEncoder(Encoder):
    """
        FrameEncoder base class.
        This encoders encode single frames of depth data.
        Implemented encoders implement the encode() method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, data, *args, **kwargs):
        if data.ndim > 2:
            raise ValueError(f"FrameEncoder only supports single frames, you provided: {data.shape}")
        return super().encode(data, *args, **kwargs)

class VideoEncoder(Encoder):
    """
        VideoEncoder base class.
        This encoders encode sequences of frames (videos) of depth data.
        Implemented encoders implement the encode() method.

        Arguments:
        - frame_encoder: An instance of FrameEncoder to encode individual frames.
                         This should only be used if the VideoEncoder is used as a wrapper for a FrameEncoder.
    """


    def __init__(self, 
                  frame_encoder:FrameEncoder = None,
                  *args, **kwargs):
        
        self.frame_encoder = frame_encoder
        super().__init__(*args, **kwargs)

    def encode(self, data, *args, **kwargs):
        """
            Encoding function that serves as a wrapper for the FrameEncoder.
            If the VideoEncoder serves as a base class, overwrite this method in the subclass.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...] 

        # Check whether this is called 
        if type(self) is VideoEncoder:
            assert self.frame_encoder is not None, "VideoEncoder requires a FrameEncoder instance."
            compressed_data = []

            for i in range(data.shape[0]):
                compressed_frame = self.frame_encoder.encode(data[i], *args, **kwargs)
                compressed_data.append(compressed_frame)
            
            return compressed_data
        
        raise NotImplementedError("Subclasses should implement the encode() method.")
    
############################################
########## Implemented Encoders ############
############################################    

############### Raw Encoder ################
class EncoderRaw(FrameEncoder):
    """
        Frame encoder converting the input data to the raw bytes of a numpy array.
    
    """
    name: str = "raw"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        buf = io.BytesIO()
        np.save(buf, data)
        dbytes = buf.getvalue()
        buf.close()
        return dbytes

class EncoderRawVideo(VideoEncoder):
    """
        Video encoder converting the input data to the raw bytes of a numpy array.
        Wraps the corresponding frame encoder for sequences of depth frames.
    """
    name: str = "raw"

    def __init__(self, *args, **kwargs):
        super().__init__(frame_encoder=EncoderRaw(), *args, **kwargs)

    def encode(self, data: np.ndarray, *args, **kwargs) -> List[bytes]:
        if data.ndim == 2:
            data = data[np.newaxis, ...] 

        compressed_data = []
        for i in range(data.shape[0]):
            frame_compressed = self.frame_encoder.encode(data[i], *args, **kwargs)
            compressed_data.append(frame_compressed)
        return compressed_data


############## TRVL Encoder ################
class EncoderTRVL(FrameEncoder):
    """
        Frame encoder for the temporal RVL encoding.

        Reference: https://github.com/hanseuljun/temporal-rvl
        C++ binding: ./backend/cpp/include/trvl.h
    
    """
    name: str = "TRVL"
    def __init__(self, 
                frame_size: int,
                change_threshold:int = 10,
                invalidation_threshold: int = 2,
                suppress_warnings: bool = False, *args, **kwargs
                ):
        self.frame_size = frame_size
        self.change_threshold = change_threshold
        self.invalidation_threshold = invalidation_threshold
        self.suppress_warnings = suppress_warnings
        if _EncoderTRVL is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        self._encoder = _EncoderTRVL(frame_size, change_threshold, invalidation_threshold)


    def encode(self, data: np.ndarray, keyframe: bool=True, *args, **kwargs) -> bytes:
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).flatten().tolist()
        #data_shape = data.shape
        data_compressed = self._encoder.encode(data, keyframe)
        return data_compressed
    
class EncoderTRVLVideo(VideoEncoder):
    """
        Video encoder for the temporal RVL encoding.
        Wraps the corresponding frame encoder for sequeunces of depth frames.
    """
    name: str = "TRVL"
    def __init__(self, 
                frame_size: int,
                change_threshold:int = 10,
                invalidation_threshold: int = 2,
                keyframe_interval: int = 10,
                suppress_warnings: bool = False, *args, **kwargs
                ):
        self.keyframe_interval = keyframe_interval
        self.suppress_warnings = suppress_warnings
        frame_encoder = EncoderTRVL(frame_size, change_threshold, invalidation_threshold, suppress_warnings=False)
        super().__init__(frame_encoder=frame_encoder)


    def encode(self, data, *args, **kwargs):
        if data.ndim == 2:
            data = data[np.newaxis, ...] 

        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)

        compressed_data = []
        keyframes = []
        for i in range(data.shape[0]):
            is_keyframe = (i % self.keyframe_interval == 0) if self.keyframe_interval > 0 else True
            if is_keyframe: keyframes.append(i)

            compressed_frame = self.frame_encoder.encode(data[i], is_keyframe, *args, **kwargs)
            compressed_data.append(compressed_frame)
        
        return compressed_data, keyframes

############### RVL Encoder ################
class EncoderRVL(FrameEncoder):
    """
        Frame encoder for the RVL encoding.
        
        
        Reference: https://github.com/hanseuljun/temporal-rvl
        C++ binding: ./backend/cpp/include/rvl.h
    """
    name: str = "RVL"
    def __init__(self, frame_size: int, suppress_warnings: bool = False, *args, **kwargs):
        self.frame_size = frame_size
        self.suppress_warnings = suppress_warnings

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).flatten().tolist()
        if RVLCompress is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        data_compressed = RVLCompress(data)
        return data_compressed

class EncoderRVLVideo(VideoEncoder):
    name: str = "RVL"

    def __init__(self, frame_size: int, suppress_warnings: bool = False, *args, **kwargs):
        self.suppress_warnings = suppress_warnings
        frame_encoder = EncoderRVL(frame_size, suppress_warnings=False)
        super().__init__(frame_encoder=frame_encoder)

    def encode(self, data: np.ndarray, *args, **kwargs):
        if data.ndim == 2:
            data = data[np.newaxis, ...] 
        
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)

        compressed_data = []
        for i in range(data.shape[0]):
            compressed_frame = self.frame_encoder.encode(data[i], *args, **kwargs)
            compressed_data.append(compressed_frame)
        
        return compressed_data