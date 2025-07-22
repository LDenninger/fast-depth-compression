import numpy as np
from typing import List

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
    
    def __init__(self, frame_size: int = None, *args, **kwargs):
        self._frame_size = frame_size

    def __call__(self, data: bytes, frame_size: int = None, *args, **kwargs) -> np.ndarray:
        self.frame_size = frame_size
        return self.decode(data, *args, **kwargs)

    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray:
        """
            The decoding function to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement the decode() method.")

class FrameDecoder(Decoder):
    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray:
        return super().decode(data, *args, **kwargs)
    
    @property
    def frame_size(self):
        return self._frame_size
    
    @frame_size.setter
    def frame_size(self, value):
        self._frame_size = value

class VideoDecoder(Decoder):
    def __init__(self, frame_decoder: FrameDecoder = None, *args, **kwargs):
        frame_size = kwargs.get('frame_size', None)
        self.frame_decoder = frame_decoder
        if self.frame_decoder is not None and frame_size is None:
            frame_size = self.frame_decoder.frame_size
        super().__init__(frame_size=frame_size, *args, **kwargs)

    @property
    def frame_size(self):
        if self.frame_decoder is not None:
            return self.frame_decoder.frame_size
        return None
    
    @frame_size.setter
    def frame_size(self, value):
        if self.frame_decoder is not None:
            self.frame_decoder.frame_size = value

        

    def decode(self, data: List[bytes], *args, **kwargs):
        if type(self) is VideoDecoder:
            assert self.frame_decoder is not None, "VideoDecoder requires a FrameDecoder instance."
            outputs = []
            for i in range(len(data)):
                frame_decoded = self.frame_decoder.decode(data[i], *args, **kwargs)
                frame_decoded = np.array(frame_decoded, dtype=np.int16)
                outputs.append(frame_decoded)
            return outputs
        
        return super().decode(data, *args, **kwargs)

class DecoderTRVL(FrameDecoder):
    name: str = "TRVL"
    binding_class = _DecoderTRVL
    def __init__(self, 
                frame_size: int = None,
                suppress_warnings: bool = False, *args, **kwargs
                ):
        super().__init__(frame_size=frame_size, *args, **kwargs)
        self.suppress_warnings = suppress_warnings
        if _DecoderTRVL is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        
        self._decoder = None
        if frame_size is not None:
            self._decoder = _DecoderTRVL(frame_size)

    def decode(self, data: List[bytes], keyframe: bool = False, *args, **kwargs) -> np.ndarray:
        data_uncompressed = self._decoder.decode(data, keyframe)
        data_uncompressed = np.array(data_uncompressed, dtype=np.int16)
        invalid_data = np.isnan(data_uncompressed)
        data_uncompressed[invalid_data] = 0 
        return data_uncompressed
    
    @property
    def frame_size(self):
        return self._frame_size
    
    @frame_size.setter
    def frame_size(self, value):
        self._frame_size = value
        if value is not None:
            self._decoder = _DecoderTRVL(self._frame_size)

class DecoderTRVLVideo(VideoDecoder):
    name: str = "TRVL"
    def __init__(self, frame_size: int = None, suppress_warnings: bool = False, *args, **kwargs):
        frame_decoder = DecoderTRVL(frame_size, suppress_warnings)
        super().__init__(frame_decoder=frame_decoder)

    def decode(self, data: List[bytes], keyframes: list = None, *args, **kwargs) -> np.ndarray:
        if keyframes is None:
            keyframes = [0]

        keyframes = set(keyframes)

        frame_size = kwargs.get('frame_size', None)
        if frame_size is not None: self.frame_size = frame_size

        outputs = []
        for i in range(len(data)):
            is_keyframe = i in keyframes
            frame_decoded = self.frame_decoder.decode(data[i], is_keyframe)
            outputs.append(frame_decoded)

        outputs = np.stack(outputs, axis=0)
        return outputs

class DecoderRVL(FrameDecoder):
    name: str = "RVL"

    def __init__(self, frame_size: int = None, suppress_warnings: bool = False, *args, **kwargs):
        self.suppress_warnings = suppress_warnings
        super().__init__(frame_size=frame_size, *args, **kwargs)

    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray:
        if RVLDecompress is None:
            raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
        data_uncompressed = RVLDecompress(data, self._frame_size)
        return np.array(data_uncompressed, dtype=np.int16)

class DecoderRVLVideo(VideoDecoder):
    name: str = "RVL"
    def __init__(self, frame_size: int = None, suppress_warnings: bool = False, *args, **kwargs):
        self.suppress_warnings = suppress_warnings
        frame_decoder = DecoderRVL(frame_size, suppress_warnings=False)
        super().__init__(frame_decoder=frame_decoder)


    def decode(self, data: List[bytes], *args, **kwargs) -> np.ndarray:
        assert self.frame_decoder.frame_size, "Please set frame_size before decoding."

        outputs = []
        for i in range(len(data)):
            frame_decoded = self.frame_decoder.decode(data[i])
            frame_decoded = np.array(frame_decoded, dtype=np.int16)
            outputs.append(frame_decoded)

        outputs = np.stack(outputs, axis=0)
        return outputs