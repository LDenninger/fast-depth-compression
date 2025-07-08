# 🚀 Fast Depth Compression (fdcomp)

> A high-performance Python library for **lossless depth image compression** using state-of-the-art TRVL (Temporal RVL) and RVL algorithms.

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-14-orange.svg)](https://en.cppreference.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

**Disclaimer:** The project is currently under development and there might be installation issues or issues in the backend. Please report these in the issues, I am trying to fix it as soon as possible. Pre-compiled binaries will be available at some point but for now please install everything from source. There are also some logic parts which in the future will be moved into the CPP-backend in favor of efficiency.

## ✨ Features

- **🔥 Ultra-Fast Performance**: C++ backend with Python bindings for optimal speed
- **📊 TRVL Algorithm**: Advanced temporal compression for depth video sequences
- **⚡ RVL Algorithm**: Wilson's Run-Length Variable compression implementation
- **🔄 Lossless Compression**: Perfect reconstruction of depth data
- **🎯 Multiple Interfaces**: Support for both Wilson's original functions and modern C++ RAII versions
- **📦 Easy Integration**: Simple Python API for seamless workflow integration

## 🏗️ Installation

### Prerequisites

Before installation, ensure you have:

- **Python** 3.6 or higher
- **CMake** 3.15 or higher  
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

### RVL Algorithm Usage

```python
import fdcomp
import numpy as np

# Generate sample depth data
depth_data = np.random.randint(0, 32767, size=1000, dtype=np.int16)

# Compress using RVL
compressed = fdcomp.RVLCompress(depth_data.tolist())
print(f"Original: {len(depth_data) * 2} bytes → Compressed: {len(compressed)} bytes")

# Decompress
decompressed = fdcomp.RVLDecompress(compressed, len(depth_data))
print(f"Lossless: {np.array(decompressed) == depth_data}")
```

## 📖 API Reference

### 🎬 TRVL Classes (Temporal Compression)

| Class | Description | Parameters |
|-------|-------------|------------|
| `EncoderTRVL` | Temporal depth encoder | `frame_size`, `change_threshold`, `invalidation_threshold` |
| `DecoderTRVL` | Temporal depth decoder | `frame_size` |

### ⚡ RVL Functions (Single Frame Compression)

| Function | Description | Parameters |
|----------|-------------|------------|
| `RVLCompress()` | Modern RVL compression | `depth_buffer: List[int]` |
| `RVLDecompress()` | Modern RVL decompression | `compressed_data: bytes`, `num_pixels: int` |
| `CompressRVL()` | Wilson's original RVL | `depth_buffer: List[int]` |
| `DecompressRVL()` | Wilson's original RVL | `compressed_data: bytes`, `num_pixels: int` |

### 💾 File I/O Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `save()` | Save depth array to file | `data: np.ndarray`, `filename: str` |
| `load()` | Load depth array from file | `filename: str` |
| `dump()` | Serialize depth data | `data: np.ndarray` |

## 🧪 Examples

Run the included examples to see fdcomp in action:

```bash
# Navigate to examples directory
cd examples

# Run TRVL compression example
python encoding.py

# Run file save/load example  
python saving.py
```

## 🔧 Development

### Building from Source

```bash
# Clone the repository
git clone <your-repo-url>
cd fast-depth-compression

# Install in development mode
pip install -e .

# Run examples
cd examples
python encoding.py
```

### Project Structure

```
fast-depth-compression/
├── 📁 backend/              # C++ implementation
│   ├── 📁 cpp/              # Core algorithms
│   └── 📁 bindings/         # Python bindings
├── 📁 fdcomp/               # Python package
├── 📁 examples/             # Usage examples
└── 📄 README.md             # This file
```

## 🏆 Performance

fdcomp delivers exceptional compression performance:

- **Speed**: C++ backend ensures minimal latency
- **Compression**: Typically 2-10x compression ratios on depth data
- **Quality**: Mathematically lossless reconstruction
- **Memory**: Efficient streaming compression for large sequences

## 📜 Algorithm References
If you use this library in your work, please consider citing the authors of the compression algorithms used:

- **RVL**: Wilson, A. D. (2017). "Fast lossless depth image compression." *ACM International Conference on Interactive Surfaces and Spaces*
- **TRVL**: Jun, H & Bailenson, J. (2020). "Temporal RVL: A Depth Stream Compression Method"

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and enhancement requests.

## 📄 License

[Your License Here]

---

**Made with ❤️ for the computer vision community**