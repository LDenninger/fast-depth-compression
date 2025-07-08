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

class Encoder:
    def __call__(self, data: np.ndarray, *args, **kwargs) -> bytes:
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
    
class EncoderTRVL(Encoder):
    name: str = "TRVL"
    def __init__(self, 
                frame_size: int,
                change_threshold:int = 10,
                invalidation_threshold: int = 2,
                suppress_warnings: bool = False
                ):
        self.frame_size = frame_size
        self.change_threshold = change_threshold
        self.invalidation_threshold = invalidation_threshold
        self.suppress_warnings = suppress_warnings
        if _EncoderTRVL is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        self._encoder = _EncoderTRVL(frame_size, change_threshold, invalidation_threshold)


    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).flatten().tolist()
        #data_shape = data.shape
        data_compressed = self._encoder.encode(data)
        return data_compressed
    
class EncoderRVL(Encoder):
    name: str = "RVL"

    def __init__(self, frame_size: int, suppress_warnings: bool = False):
        self.frame_size = frame_size
        self.suppress_warnings = suppress_warnings

    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        data = np.ascontiguousarray(data).flatten().tolist()
        if RVLCompress is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        data_compressed = RVLCompress(data)
        return data_compressed
