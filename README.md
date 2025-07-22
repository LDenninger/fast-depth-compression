# 🚀 Fast Depth Compression (fdcomp)

> A high-performance Python library for **lossless depth image compression** using state-of-the-art TRVL (Temporal RVL) and RVL algorithms.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-14-orange.svg)](https://en.cppreference.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

**Disclaimer:** The project is currently under development and there might be installation issues or issues in the backend. Please report these in the issues, I am trying to fix it as soon as possible. Pre-compiled binaries will be available at some point but for now please install everything from source. There are also some logic parts which in the future will be moved into the CPP-backend in favor of efficiency.

## ✨ Features

- **🔥 Ultra-Fast Performance**: C++ backend with Python bindings for optimal speed
- **📊 TRVL Algorithm**: Advanced temporal compression for depth video sequences
- **⚡ RVL Algorithm**: Wilson's Run-Length Variable compression implementation
- **🔄 Lossless Compression**: Perfect reconstruction of depth data
- **📦 Easy Integration**: Simple Python API for seamless workflow integration

## 📚 Table of Contents

| Section                                                          | Description                                  |
| ---------------------------------------------------------------- | -------------------------------------------- |
| [📁 **Project Structure**](#project-structure)                   | Overview of project folders and files        |
| [⚙️ **Installation**](#installation)                              | Prerequisites & installation steps           |
| [🚀 **Quick Start**](#quick-start)                               | Basic usage examples                         |
| [📖 **API Reference**](#api-reference)                           | Details of encoders, decoders, utilities     |
| [🧪 **Tests**](#tests)                                           | How to run the test suite                    |
| [📜 **Algorithm References**](#algorithm-references)             | Citations for compression algorithms         |
| [🤝 **Contributing**](#contributing)                             | Guidelines to contribute to the project      |
| [📄 **License**](#license)                                       | Licensing information                        |

<a name="project-structure"></a>
## 📁 Project Structure

```
fast-depth-compression/
├── 📁 backend/              # C++ implementation
│   ├── 📁 cpp/              # Core algorithms
│   └── 📁 bindings/         # Python bindings
├── 📁 fdcomp/               # Python package
├── 📁 examples/             # Usage examples
└── 📄 README.md             # This file
```

<a name="installation"></a>
## 🏗️ Installation

### Prerequisites

These prereuisities are currently based on my current setups, I have tested the package on.

So, before installation, ensure you have:

- **Python** 3.11 or higher
- **CMake** 3.30 or higher  
- **OpenCV** development libraries
- **C++14** compatible compiler

### 📦 Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libopencv-dev cmake build-essential

# macOS with Homebrew
brew install opencv cmake

# Windows with vcpkg
vcpkg install opencv
```

### 🚀 Install fdcomp

#### Option 1: Direct Installation (Recommended)
```bash
pip install .
```

#### Option 2: Development Installation
```bash
pip install -e .
```

#### Option 3: With Build Dependencies
```bash
pip install scikit-build
pip install .
```

<a name="quick-start"></a>
## 🎯 Quick Start

### Basic TRVL Compression Example

```python
import fdcomp
import numpy as np
import matplotlib.pyplot as plt

# Load depth data
depth_arr = np.load("examples/depth.npz")['depth'][0]
height, width = depth_arr.shape

# Initialize TRVL encoder/decoder
encoder = fdcomp.EncoderTRVL(
    frame_size=width * height, 
    change_threshold=10, 
    invalidation_threshold=2
)
decoder = fdcomp.DecoderTRVL(frame_size=width * height)

# Compress depth frame
compressed_data = encoder.encode(depth_arr, keyframe=False)
print(f"Original size: {depth_arr.nbytes} bytes")
print(f"Compressed size: {len(compressed_data)} bytes")
print(f"Compression ratio: {depth_arr.nbytes / len(compressed_data):.2f}x")

# Decompress and verify
decompressed = decoder.decode(compressed_data)
decompressed = np.reshape(decompressed, (height, width))
decompressed = decompressed.view(np.float16)

# Check lossless compression
l2_error = np.linalg.norm(depth_arr - decompressed)
print(f"L2 reconstruction error: {l2_error}")  # Should be 0.0 for lossless

# Save visualization
original_img = ((depth_arr.astype(np.float32) / depth_arr.max()) * 255).astype(np.uint8)
restored_img = ((decompressed.astype(np.float32) / decompressed.max()) * 255).astype(np.uint8)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Depth')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(restored_img, cmap='gray')
plt.title('Decompressed Depth')
plt.axis('off')

plt.tight_layout()
plt.savefig('compression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simple File Save/Load Example

```python
import fdcomp
import numpy as np

# Load your depth data
depth_arr = np.load("examples/depth.npz")['depth'][0]

# Save compressed depth file
fdcomp.save(depth_arr, "my_depth.dep")

# Load and verify
loaded_depth = fdcomp.load("my_depth.dep")

print(f"Original shape: {depth_arr.shape}, dtype: {depth_arr.dtype}")
print(f"Loaded shape: {loaded_depth.shape}, dtype: {loaded_depth.dtype}")

# Verify lossless compression
l2_error = np.linalg.norm(depth_arr - loaded_depth)
print(f"Reconstruction error: {l2_error}")  # Should be 0.0
```

<a name="api-reference"></a>
## 📖 API Reference

### Base Classes

#### Encoders

**Class: Encoder**
_Abstract base class for all encoders_

```python
class Encoder:
    def __call__(self, data: np.ndarray, *args, **kwargs) -> bytes
    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes
    def _cast_int16(self, data: np.ndarray, suppress_warnings: bool=False) -> np.ndarray
```

- ``__call__(data, *args, **kwargs) → bytes``
  Invokes the encoder by forwarding to `encode()`.
- ``encode(data, *args, **kwargs) → bytes``
  **Abstract**: implement this in subclasses.
- ``_cast_int16(data, suppress_warnings=False) → np.ndarray``
  Casts input array to int16, with optional warnings.

**Class: FrameEncoder**
_Inherits from Encoder; validates single-frame (2D) inputs_

```python
class FrameEncoder(Encoder):
    def encode(self, data: np.ndarray, *args, **kwargs) -> bytes
```

- **encode(data, *args, **kwargs) → bytes**
  Ensures `data.ndim == 2` before encoding.

**Class: VideoEncoder**
_Inherits from Encoder; wraps a FrameEncoder for multi-frame support_

```python
class VideoEncoder(Encoder):
    def __init__(self, frame_encoder: FrameEncoder = None)
    def encode(self, data: np.ndarray, *args, **kwargs) -> List[bytes]
```

- ``__init__(frame_encoder: FrameEncoder = None)``
  Optionally accepts a `FrameEncoder` instance.
- ``encode(data, *args, **kwargs) → List[bytes]``
  Handles 2D or 3D arrays, delegating per-frame encoding to `frame_encoder`.

#### Decoders

**Class: Decoder**
_Abstract base class for all decoders_

```python
class Decoder:
    def __call__(self, data: bytes or List[bytes], frame_size: int = None, *args, **kwargs) -> np.ndarray
    def decode(self, data: bytes or List[bytes], *args, **kwargs) -> np.ndarray
```

- ``__call__(data, frame_size=None, *args, **kwargs) → np.ndarray``
  Initializes `frame_size` and delegates to `decode()`.
- ``decode(data, *args, **kwargs) → np.ndarray``
  **Abstract**: implement this in subclasses.

**Class: FrameDecoder**
_Inherits from Decoder; validates single-frame (bytes) inputs_

```python
class FrameDecoder(Decoder):
    def decode(self, data: bytes, *args, **kwargs) -> np.ndarray
```

- ``decode(data, *args, **kwargs) → np.ndarray``
  Ensures per-frame decoding logic is applied correctly.

**Class: VideoDecoder**
_Inherits from Decoder; wraps a FrameDecoder for multi-frame support_

```python
class VideoDecoder(Decoder):
    def __init__(self, frame_decoder: FrameDecoder = None)
    def decode(self, data: List[bytes], *args, **kwargs) -> List[np.ndarray]
```

- ``__init__(frame_decoder: FrameDecoder = None)``
  Optionally accepts a `FrameDecoder` instance.
- ``decode(data, *args, **kwargs) → List[np.ndarray]``
  Decodes each frame via `frame_decoder` and returns a list of arrays.

### Algorithm Implementations

#### TRVL
| Class                  | Description                                    | Key Params                                      |
|------------------------|------------------------------------------------|-------------------------------------------------|
| `EncoderTRVL`          | Frame-level temporal RVL encoder               | `frame_size`, `change_threshold`, `invalidation_threshold` |
| `EncoderTRVLVideo`     | Video encoder with keyframe interval           | `frame_size`, `keyframe_interval`, ...          |
| `DecoderTRVL`          | Frame-level temporal RVL decoder               | `frame_size`                                    |
| `DecoderTRVLVideo`     | Video decoder handling keyframes               | `frame_size`, `keyframes`                       |

#### RVL
| Class                  | Description                                    | Key Params                                      |
|------------------------|------------------------------------------------|-------------------------------------------------|
| `EncoderRVL`           | Frame-level RVL encoder                        | `frame_size`                                    |
| `EncoderRVLVideo`      | Video RVL encoder                              | `frame_size`                                    |
| `DecoderRVL`           | Frame-level RVL decoder                        | `frame_size`                                    |
| `DecoderRVLVideo`      | Video RVL decoder                              | `frame_size`                                    |

### File I/O Utilities

- **save(data: np.ndarray, filename: str, encoder: Union[str, Encoder])**
- **load(filename: str, decoder: Union[str, Decoder]) → np.ndarray**
- **loads(data: Union[bytes, List[bytes]], decoder: Union[str, Decoder]) → np.ndarray**
- **dump(data: np.ndarray, encoder: Union[str, Encoder]) → bytes**

<a name="tests"></a>
## 🧪 Tests

Unit tests cover all core algorithms and I/O functions, ensuring lossless compression across both TRVL and RVL implementations.

- **test_trvl.py**: Verify TRVL emporal encoder/decoder frame-wise and for videos.
- **test_rvl.py**: Verify RVL emporal encoder/decoder frame-wise and for videos.
- **test_io.py**: Verify I/O operations for TRVL and RVL algorithms.

To run all tests:

```bash
pytest -v --disable-warnings --maxfail=1
```

To run separate test with outputs:
```bash
python tests/test_trvl.py 
```



<a name="algorithm-references"></a>
## 📜 Algorithm References
If you use this library in your work, please consider citing the authors of the compression algorithms used:

- **RVL**: Wilson, A. D. (2017). "Fast lossless depth image compression." *ACM International Conference on Interactive Surfaces and Spaces*
- **TRVL**: Jun, H & Bailenson, J. (2020). "Temporal RVL: A Depth Stream Compression Method"

<a name="contributing"></a>
## 🤝 Contributing

I welcome any contributions or algorithm requests for depth compression. Provided how much work is required, meaning whether there is an existing C++ implementation or only documentation, this process might take longer.

### Upcoming Changes
There are a few things that are already planned to be added/changed:
- The complete abstraction layer and implementation of frame-wise and video encoder/decoders has to be moved to the C++ backend
- Tests exclusively for the C++ backend
- Tests to profile the encoder and decoder with comprehensive summary

<a name="license"></a>
## 📄 License

Parts of this project (`./backend/cpp/include/rvl.h`, `./backend/cpp/include/trvl.h`) are licensed under Apache 2.0, taken from [https://github.com/hanseuljun/temporal-rvl/tree/master](https://github.com/hanseuljun/temporal-rvl/tree/master)

All other parts are licensed under the **Apache 2.0** license.



