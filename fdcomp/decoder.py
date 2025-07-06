import numpy as np
import fdc_bindings as fdcb

class Decoder:
    def __call__(self, data: bytes, *args, **kwargs) -> np.ndarray:
        return self.decode(data, *args, **kwargs)

    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray:
        """
            The decoding function to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement the decode() method.")

class DecoderTRVL(Decoder):
    def __init__(self, 
                frame_size: int,
                suppress_warnings: bool = False
                ):
        self.frame_size = frame_size
        self.suppress_warnings = suppress_warnings
        self._decoder = fdcb.DecoderTRVL(frame_size)


    def encode(self, data: bytes, *args, **kwargs) -> bytes:
        data_uncompressed = self._decoder.decode(data)
        data_uncompressed = np.array(data_uncompressed, dtype=np.int16)
        return data_uncompressed

