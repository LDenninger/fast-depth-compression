import numpy as np

# Import bindings with fallback
def _get_bindings():
    try:
        from .fdc_bindings import DecoderTRVL as _DecoderTRVL, RVLDecompress
        return _DecoderTRVL, RVLDecompress
    except ImportError:
        # If importing during development without compiled bindings
        return None, None

_DecoderTRVL, RVLDecompress = _get_bindings()

class Decoder:
    def __call__(self, data: bytes, *args, **kwargs) -> np.ndarray:
        return self.decode(data, *args, **kwargs)

    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray:
        """
            The decoding function to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement the decode() method.")

class DecoderTRVL(Decoder):
    name: str = "TRVL"
    def __init__(self, 
                frame_size: int,
                suppress_warnings: bool = False
                ):
        self.frame_size = frame_size
        self.suppress_warnings = suppress_warnings
        if _DecoderTRVL is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        self._decoder = _DecoderTRVL(frame_size)


    def decode(self, data: bytes, *args, **kwargs) -> bytes:
        data_uncompressed = self._decoder.decode(data)
        data_uncompressed = np.array(data_uncompressed, dtype=np.int16)
        return data_uncompressed

class DecoderRVL(Decoder):
    name: str = "RVL"
    def __init__(self, frame_size: int, suppress_warnings: bool = False):
        self.frame_size = frame_size
        self.suppress_warnings = suppress_warnings

    def decode(self, data: bytes, *args, **kwargs) -> bytes:
        if RVLDecompress is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        data_compressed = RVLDecompress(data, self.frame_size)
        return data_compressed