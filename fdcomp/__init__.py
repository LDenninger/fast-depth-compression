# I/O Operations
from .fdcomp import load, loads, dump, save
# Encoder and Decoder Classes
from .encoder import (
    # Base Encoder classes
    Encoder, FrameEncoder, VideoEncoder,
    # TRVL
    EncoderTRVL, EncoderTRVLVideo, EncoderRVLVideo,
    # RVL
    EncoderRVL, EncoderTRVLVideo
)
from .decoder import (
    # Base Decoder classes
    Decoder, FrameDecoder, VideoDecoder,
    # TRVL
    DecoderTRVL, DecoderTRVLVideo, DecoderRVLVideo,
    # RVL
    DecoderRVL, DecoderRVLVideo
)

# Import the compiled C++ bindings at module level
try:
    from .fdc_bindings import (
        CompressRVL,
        DecompressRVL,
        RVLCompress,
        RVLDecompress
    )
    # Make them available at module level
    globals().update({
        'CompressRVL': CompressRVL,
        'DecompressRVL': DecompressRVL,
        'RVLCompress': RVLCompress,
        'RVLDecompress': RVLDecompress
    })
    _bindings_available = True
except ImportError:
    # If bindings are not available, provide dummy functions that raise meaningful errors
    def _raise_bindings_error(*args, **kwargs):
        raise ImportError("C++ bindings not available. Please install the package with 'pip install .'")
    
    CompressRVL = _raise_bindings_error
    DecompressRVL = _raise_bindings_error
    RVLCompress = _raise_bindings_error
    RVLDecompress = _raise_bindings_error
    _bindings_available = False

__all__ = [
    "load", "dump", "save",
    "Encoder", "EncoderTRVL", "EncoderRVL", 
    "Decoder", "DecoderTRVL", "DecoderRVL",
    "CompressRVL", "DecompressRVL",
    "RVLCompress", "RVLDecompress"
]