import numpy as np
import fdc_bindings as fdcb
from termcolor import colored

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
        self._encoder = fdcb.EncoderTRVL(frame_size, change_threshold, invalidation_threshold)


    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes:
        data = self._cast_int16(data, suppress_warnings=self.suppress_warnings)
        #data_shape = data.shape
        data_compressed = self._encoder.encode(data.flatten().tolist())
        return data_compressed

