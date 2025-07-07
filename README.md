# Fast Depth Compression

## Installation

### Source 

First, build the cpp backend from source:
```bash
mkdir build
cd build
cmake ../backend -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

For now, export the python bindings manually to the python path:
```bash
export PYTHONPATH=$PWD/bindings:$PYTHONPATH
```

Then install the python package:
```bash
cd ..
pip install .
```