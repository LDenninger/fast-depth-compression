# 🚀 Fast Depth Compression (fdcomp)

> A high-performance Python library for **lossless depth image compression** using state-of-the-art TRVL (Temporal RVL) and RVL algorithms.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-14-orange.svg)](https://en.cppreference.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/github/actions/workflow/status/LDenninger/fast-depth-compression/package-test.yml?label=Tests&style=flat)](https://github.com/LDenninger/fast-depth-compression/actions/workflows/package-test.yml)


**Disclaimer:** The project is currently under development and there might be installation issues or issues in the backend. Please report these in the issues, I am trying to fix it as soon as possible. Pre-compiled binaries will be available at some point but for now please install everything from source.

**Recent Major Updates:**
- ✅ **Backend Migration Complete**: The Python frontend has been successfully moved to the C++ backend for improved performance
- ✅ **Enhanced C++ Bindings**: Optimized Python bindings with zero-copy operations and NumPy integration
- ✅ **Performance Testing**: Added comprehensive performance benchmarking tools
- ✅ **Improved API**: Streamlined encoder/decoder interfaces with better error handling

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
| [📈 **Performance**](#performance)                   | Overview of project folders and files        |
| [⚙️ **Installation**](#installation)                              | Prerequisites & installation steps          |
| [🚀 **Quick Start**](#quick-start)                               | Basic usage examples                         |
| [📖 **API Reference**](#api-reference)                           | Details of encoders, decoders, utilities     |
| [🧪 **Tests**](#tests)                                           | How to run the test suite                    |
| [📜 **Algorithm References**](#algorithm-references)             | Citations for compression algorithms         |
| [📄 **License**](#license)                                       | Licensing information                        |

<a name="project-structure"></a>
## 📁 Project Structure

```
fast-depth-compression/
├── 📁 backend/              # C++ implementation
│   ├── 📁 cpp/              # Core algorithms
│   └── 📁 bindings/         # Python bindings
├── 📁 fdcomp/               # Python package
├── 📁 examples/             # Usage/data examples
├── 📁 tests/                # Tests for Python/C++
└── 📄 README.md             # This file
```

<a name="performance"></a>
## 📈 Performance

- **Easy Python API**: Shallow Numpy API with similar structure to `json` and `yaml` libraries.
- **High-performance C++ core**: Compute-heavy paths implemented in modern C++.
- **Zero-copy bridging**: NumPy arrays are passed to C++ without intermediate copies.
- **GIL released**: Encode/decode run outside the Python GIL, enabling real multi-threaded 

| Algorithm      | Video Length   | Shape          | Loading      | Saving       |
|----------------|----------------|----------------|----------------|----------------|
| TRVL           | 139            | (350, 630)  | 129.19 ms  | 165.75 ms  |

Note that these were measured in Python with much of the running time coming from the bindings.

<a name="installation"></a>
## 🏗️ Installation

### Prerequisites

These prereuisities are currently based on my current setups, I have tested the package on.
All tests were made on *Ubuntu 22.4*

So, before installation, ensure you have:

- **Python** 3.11 or higher
- **CMake** 3.30 or higher  
- **OpenCV** development libraries
- **C++14** compatible compiler

### 📦 Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake build-essential
# (Optional) If you want to build tests
sudo apt-get install libopencv-dev 
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

#### Option 3: Build C++ backend with tests
```bash
mkdir build
cd build
cmake -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_CPP_TESTS=ON ..
cmake --build .
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

The fdcomp API consists of a high-level Python interface powered by an optimized C++ backend with zero-copy operations and NumPy integration.

### 🐍 Python API

#### High-Level Functions

``save(data: np.ndarray, file: Union[str, Path], encoder: Union[str, Encoder] = "trvl", **kwargs)``
- Save depth data to compressed file format
- **data**: NumPy array with depth data (2D or 3D)
- **file**: Output file path (automatically adds .dep extension)
- **encoder**: Compression algorithm - "trvl", "rvl", or "raw"

`load(file: Union[str, Path], decoder: Union[str, Decoder] = None, **kwargs) → np.ndarray`
- Load compressed depth data from file
- **file**: Input file path
- **decoder**: Decompression algorithm (auto-detected if None)
- **Returns**: Reconstructed depth data as NumPy array

``loads(data: Union[bytes, List[bytes]], decoder: Union[str, Decoder], frame_size: int, **kwargs) → np.ndarray``
- Load depth data from raw compressed bytes
- **data**: Compressed bytes or list of frame bytes
- **decoder**: Decompression algorithm
- **frame_size**: Number of pixels per frame

``dump(data: np.ndarray, encoder: Union[str, Encoder], **kwargs) → Union[bytes, List[bytes]]``
- Compress depth data to bytes without saving to file
- **data**: NumPy array with depth data
- **encoder**: Compression algorithm

``inspect(file: Union[str, Path], print_result: bool = True) → dict``
- Analyze compressed file metadata
- **Returns**: Dictionary with file information (shape, dtype, compression type, etc.)

#### Base Classes

**Class: Encoder**
_C++ backend wrapper for all encoders with optimized NumPy integration_

```python
class Encoder:
    def encode(self, data: np.ndarray, verbose: bool = False, *args, **kwargs) → Union[bytes, List[bytes]]
    def _cast_int16(self, data: np.ndarray, suppress_warnings: bool = True) → np.ndarray
```

- **encode(data, verbose=False)**: Main encoding method with automatic dtype conversion
- **_cast_int16(data)**: Smart conversion to int16 with minimal copying (float16→view, float32→narrow+view)

**Class: FrameEncoder(Encoder)**
_Single-frame encoder with C++ backend_

```python
class FrameEncoder(Encoder):
    def __init__(self, frame_size: int, suppress_warnings: bool = True)
```

**Class: VideoEncoder(Encoder)**
_Multi-frame encoder with C++ backend_

```python
class VideoEncoder(Encoder):
    def __init__(self, frame_encoder: FrameEncoder, suppress_warnings: bool = True)
    def encode(self, data: np.ndarray, *args, **kwargs) → List[bytes]
```

**Class: Decoder**
_C++ backend wrapper for all decoders with NumPy output optimization_

```python
class Decoder:
    def decode(self, data: Union[bytes, List[bytes]], output_size: Tuple[int,int] = None, 
               dtype = np.int16, verbose: bool = False, *args, **kwargs) → np.ndarray
```

- **decode()**: Main decoding method returning NumPy arrays directly from C++
- **output_size**: Optional reshaping to (height, width)
- **dtype**: Output data type (int16, float16, float32)
- **verbose**: Enable timing information

**Class: FrameDecoder(Decoder)**
_Single-frame decoder with C++ backend_

**Class: VideoDecoder(Decoder)**
_Multi-frame decoder with C++ backend_

#### Algorithm Implementations

**TRVL (Temporal RVL) - Optimized for depth video sequences**

| Class                  | Description                                    | Key Parameters                                  |
|------------------------|------------------------------------------------|-------------------------------------------------|
| `EncoderTRVL`          | Frame-level TRVL encoder                       | `frame_size`, `change_threshold=10`, `invalidation_threshold=2` |
| `EncoderTRVLVideo`     | Video TRVL encoder with keyframe support      | `frame_size`, `keyframe_interval=10`, `change_threshold=10`, `invalidation_threshold=2` |
| `DecoderTRVL`          | Frame-level TRVL decoder                       | `frame_size` |
| `DecoderTRVLVideo`     | Video TRVL decoder with keyframe handling     | `frame_size`, `keyframe_interval=10` |

**RVL (Run-Length Variable) - Fast general-purpose compression**

| Class                  | Description                                    | Key Parameters                                  |
|------------------------|------------------------------------------------|-------------------------------------------------|
| `EncoderRVL`           | Frame-level RVL encoder                        | `frame_size` |
| `EncoderRVLVideo`      | Video RVL encoder                              | `frame_size` |
| `DecoderRVL`           | Frame-level RVL decoder                        | `frame_size` |
| `DecoderRVLVideo`      | Video RVL decoder with optimized flat output  | `frame_size` |

**Raw - Uncompressed NumPy serialization**

| Class                  | Description                                    |
|------------------------|------------------------------------------------|
| `EncoderRaw`           | NumPy array serialization                     |
| `EncoderRawVideo`      | Multi-frame NumPy serialization               |
| `DecoderRaw`           | NumPy array deserialization                   |
| `DecoderRawVideo`      | Multi-frame NumPy deserialization             |

### ⚡ C++ Backend API

The C++ backend provides high-performance implementations with zero-copy operations and optimized memory management.

#### Base Classes

**FrameEncoder**
```cpp
class FrameEncoder {
public:
    explicit FrameEncoder(int frame_size);
    virtual std::vector<char> encode(short* depth_buffer) = 0;
    
    int getFrameSize();
    void setFrameSize(int frame_size);
};
```

**VideoEncoder**
```cpp
class VideoEncoder {
public:
    VideoEncoder(FrameEncoder* encoder);
    
    std::vector<std::vector<char>> encode(short* depth_buffer, int num_frames);
    std::vector<std::vector<char>> encode(short* depth_buffer);
    
    void setFrameEncoder(FrameEncoder* encoder);
    void setFrameSize(int frame_size);
    void setNumFrames(int num_frames);
    int getFrameSize();
    int getNumFrames();
};
```

**FrameDecoder**
```cpp
class FrameDecoder {
public:
    explicit FrameDecoder(int frame_size);
    virtual std::vector<short> decode(char* compressed_bytes) = 0;
    
    int getFrameSize();
    void setFrameSize(int frame_size);
};
```

**VideoDecoder**
```cpp
class VideoDecoder {
public:
    explicit VideoDecoder(FrameDecoder* decoder);
    
    std::vector<std::vector<short>> decode(std::vector<char*> video_bytes);
    std::vector<std::vector<short>> decode(std::vector<std::vector<char>> video_bytes);
    
    void setFrameDecoder(FrameDecoder* decoder);
    void setFrameSize(int frame_size);
    void setNumFrames(int num_frames);
    int getFrameSize();
    int getNumFrames();
};
```

#### TRVL Implementation

**trvl::EncoderTRVL**
```cpp
class EncoderTRVL : public FrameEncoder {
public:
    EncoderTRVL(int frame_size, short change_threshold, int invalidation_threshold);
    
    std::vector<char> encode(short* depth_buffer, bool keyframe);
    std::vector<char> encode(short* depth_buffer) override;
    void setKeyframe(bool is_keyframe);
};
```

**trvl::VideoEncoderTRVL**
```cpp
class VideoEncoderTRVL : public VideoEncoder {
public:
    VideoEncoderTRVL(int keyframe_interval, int frame_size, 
                     short change_threshold, int invalidation_threshold);
    
    std::vector<std::vector<char>> encode(short* depth_buffer) override;
};
```

**trvl::DecoderTRVL**
```cpp
class DecoderTRVL : public FrameDecoder {
public:
    explicit DecoderTRVL(int frame_size);
    
    std::vector<short> decode(char* trvl_frame, bool keyframe);
    std::vector<short> decode(char* compressed_bytes) override;
};
```

**trvl::VideoDecoderTRVL**
```cpp
class VideoDecoderTRVL : public VideoDecoder {
public:
    VideoDecoderTRVL(int keyframe_interval, int frame_size);
    
    std::vector<short> decode(std::vector<char*>& video_bytes, std::vector<int> keyframes);
    std::vector<std::vector<short>> decode(std::vector<char*>& video_bytes) override;
    void setKeyframeInterval(int interval);
};
```

#### RVL Implementation

**EncoderRVL**
```cpp
class EncoderRVL : public FrameEncoder {
public:
    explicit EncoderRVL(int frame_size);
    std::vector<char> encode(short* depth_buffer) override;
};
```

**VideoEncoderRVL**
```cpp
class VideoEncoderRVL : public VideoEncoder {
public:
    explicit VideoEncoderRVL(int frame_size);
};
```

**DecoderRVL**
```cpp
class DecoderRVL : public FrameDecoder {
public:
    explicit DecoderRVL(int frame_size);
    std::vector<short> decode(char* compressed_bytes) override;
};
```

**VideoDecoderRVL**
```cpp
class VideoDecoderRVL : public VideoDecoder {
public:
    explicit VideoDecoderRVL(int frame_size);
    
    std::vector<std::vector<short>> decode(std::vector<char*>& video_bytes) override;
    std::vector<int16_t> decode_flat(std::vector<char*>& video_bytes, int& elems_per_frame_out);
};
```

## 📜 Algorithm References
If you use this library in your work, please consider citing the authors of the compression algorithms used:

- **RVL Algorithm**: Based on Wilson, A. D. (2017). "Fast lossless depth image compression." The original RVL implementation (files `backend/cpp/include/rvl.h`) is licensed under Apache 2.0.

- **TRVL Algorithm**: Based on Jun, H & Bailenson, J. (2020). "Temporal RVL: A Depth Stream Compression Method." The original TRVL implementation (files `backend/cpp/include/trvl.h`) is licensed under Apache 2.0, derived from [https://github.com/hanseuljun/temporal-rvl](https://github.com/hanseuljun/temporal-rvl).


<a name="license"></a>
## 📄 License

This project is licensed under the **Apache License 2.0**.