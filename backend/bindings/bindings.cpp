// ============================================================================
// fdc_bindings.cpp
//
// Python bindings for the fdcomp C++ library (RVL/TRVL frame & video codecs).
//
// Changes in this version: **comments and Python docstrings only**.
//   - Added comprehensive C++ comments explaining ownership, lifetime,
//     zero-copy behavior, and GIL management.
//   - Added/expanded Python-side docstrings for classes and methods to make
//     usage clear from Python (argument types, shapes, return values).
//
// NOTE: No logic or behavior has been modified. All executable statements,
//       signatures, and control flow remain exactly the same.
// ============================================================================

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rvl.h"
#include "trvl.h"

namespace py = pybind11;

// ------------------------ Helpers ------------------------
//
// The helper functions below construct NumPy arrays that **own** C++ heap-
// allocated std::vector<int16_t> storage. We move the vector into a new heap-
// allocated std::vector, then attach a `py::capsule` deleter so that memory is
// freed exactly once when the NumPy array is garbage-collected in Python.
//
// This avoids copying while still giving Python a safe, self-contained array.
//

// Create a 1D numpy array (int16) that takes ownership of a moved vector.
// Ownership is transferred via a capsule; no data is copied.
static inline py::array numpy_from_owned_vector_1d(std::vector<int16_t> &&vec) {
    auto *vec_ptr = new std::vector<int16_t>(std::move(vec));
    int16_t *data_ptr = vec_ptr->data();
    const ssize_t n = static_cast<ssize_t>(vec_ptr->size());

    // Capsule ensures the heap-allocated vector is deleted when the array dies.
    py::capsule free_when_done(vec_ptr, [](void *p) {
        delete static_cast<std::vector<int16_t> *>(p);
    });

    return py::array(
        py::buffer_info(
            data_ptr,                                 // ptr
            sizeof(int16_t),                          // itemsize
            py::format_descriptor<int16_t>::format(),// format
            1,                                        // ndim
            std::vector<ssize_t>{n},                  // shape
            std::vector<ssize_t>{static_cast<ssize_t>(sizeof(int16_t))} // strides
        ),
        free_when_done
    );
}

// Create a 2D numpy array (int16) with (rows, cols) that owns moved vector data.
// Strides are set for a C-contiguous layout.
static inline py::array numpy_from_owned_vector_2d(std::vector<int16_t> &&vec,
                                                   ssize_t rows,
                                                   ssize_t cols) {
    auto *vec_ptr = new std::vector<int16_t>(std::move(vec));
    int16_t *data_ptr = vec_ptr->data();

    // Capsule for safe lifetime management on the Python side.
    py::capsule free_when_done(vec_ptr, [](void *p) {
        delete static_cast<std::vector<int16_t> *>(p);
    });

    const ssize_t stride0 = cols * static_cast<ssize_t>(sizeof(int16_t));
    const ssize_t stride1 = static_cast<ssize_t>(sizeof(int16_t));

    return py::array(
        py::buffer_info(
            data_ptr,
            sizeof(int16_t),
            py::format_descriptor<int16_t>::format(),
            2,
            std::vector<ssize_t>{rows, cols},
            std::vector<ssize_t>{stride0, stride1}
        ),
        free_when_done
    );
}

// Convert a Python sequence of bytes-like frames into a vector of raw char*
// pointers without copying the bytes. We keep `py::bytes` objects alive in
// `keepalive` so their underlying buffers remain valid during decode.
template <typename Seq>
static inline void bytes_sequence_to_ptrs_with_keepalive(const Seq &frames_bytes,
                                                         std::vector<py::bytes> &keepalive,
                                                         std::vector<char *> &ptrs) {
    const ssize_t n = py::len(frames_bytes);
    keepalive.reserve(static_cast<size_t>(n));
    ptrs.reserve(static_cast<size_t>(n));
    for (py::handle h : frames_bytes) {
        // Cast each element to py::bytes; no data copy is performed.
        py::bytes b = py::cast<py::bytes>(h);
        keepalive.emplace_back(b); // extend lifetime
        // PyBytes_AS_STRING gives a borrowed raw pointer to the internal buffer.
        char *p = const_cast<char *>(PyBytes_AS_STRING(b.ptr()));
        ptrs.push_back(p);
    }
}

// ------------------------ Module ------------------------

PYBIND11_MODULE(fdc_bindings, m) {
    // High-level module docstring available as `fdc_bindings.__doc__` in Python.
    m.doc() = R"doc(
        Python bindings for the fdcomp C++ library.

        This module exposes efficient frame and video encoders/decoders for int16 depth
        buffers using RVL and TRVL codecs. Bindings minimize copies:
        - Encoding accepts `list/array('h')` or NumPy `int16` buffers.
        - Decoding returns NumPy `int16` arrays and uses capsules to manage lifetime.
        - Heavy compute runs with the GIL released for better multi-threading.

        Classes:
        - FrameEncoder / VideoEncoder / FrameDecoder / VideoDecoder (base)
        - EncoderTRVL / VideoEncoderTRVL / DecoderTRVL / VideoDecoderTRVL
        - EncoderRVL / VideoEncoderRVL / DecoderRVL / VideoDecoderRVL
    )doc";

    // ===== Base FrameEncoder binding =====
    py::class_<FrameEncoder>(m, "FrameEncoder", R"doc(
            Base frame encoder interface.

            Methods
            -------
            encode(depth_buffer : List[int] | array('h')) -> bytes
                Encode a single frame from a contiguous int16 buffer.
            encode(depth_buffer_np : numpy.ndarray[int16]) -> bytes
                Encode a single frame from a 1D C-contiguous NumPy int16 array.
            getFrameSize() -> int
                Return the current frame size (elements per frame).
            setFrameSize(size : int) -> None
                Set the frame size (elements per frame).
        )doc")
        .def("encode",
             [](FrameEncoder &self, std::vector<short> &buf) {
                 // Release the GIL while running the native encoder.
                 std::vector<char> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 // Return as Python bytes without extra copy.
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"),
             R"doc(
                Encode a single frame.

                Parameters
                ----------
                depth_buffer : Sequence[int] (short)
                    Flat, contiguous int16 depth buffer (elements = frame_size).

                Returns
                -------
                bytes
                    Compressed frame bytes.
            )doc")
        .def("encode",
             [](FrameEncoder &self, py::array_t<int16_t, py::array::c_style> arr) {
                 if (arr.ndim() != 1) {
                     throw std::runtime_error("FrameEncoder.encode expects a 1D int16 C-contiguous array.");
                 }
                 const int16_t *ptr = arr.data();
                 std::vector<char> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(const_cast<short*>(reinterpret_cast<const short*>(ptr)));
                 }
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer_np"),
             R"doc(
                Encode a single frame from a NumPy array.

                Parameters
                ----------
                depth_buffer_np : numpy.ndarray[int16], shape (N,), C-contiguous
                    Flat int16 depth buffer (elements = frame_size).

                Returns
                -------
                bytes
                    Compressed frame bytes.
            )doc")
        .def("getFrameSize", &FrameEncoder::getFrameSize,
             R"doc(Return the number of int16 elements expected per frame.)doc")
        .def("setFrameSize", &FrameEncoder::setFrameSize,
             py::arg("size"),
             R"doc(Set the number of int16 elements expected per frame.)doc");

    // ===== Base VideoEncoder binding =====
    py::class_<VideoEncoder>(m, "VideoEncoder", R"doc(
            Base video encoder interface that wraps a FrameEncoder.

            Construct with an existing FrameEncoder instance. The video encoder will split
            a flat buffer into frames or accept an ND NumPy array with the first dimension
            being the number of frames.
        )doc")
        .def(py::init<FrameEncoder*>(), py::arg("frame_encoder"),
             R"doc(Create a VideoEncoder bound to a FrameEncoder instance.)doc")
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf, int num_frames) {
                 // Configure number of frames; any per-frame partitioning is handled internally.
                 self.setNumFrames(num_frames);
                 std::vector<std::vector<char>> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 // Package each frame's bytes into a Python list.
                 py::list out(comp.size());
                 for (size_t i = 0; i < comp.size(); ++i) {
                     const auto &frame = comp[i];
                     out[i] = py::bytes(frame.data(), frame.size());
                 }
                 return out;
             },
             py::arg("depth_buffer"), py::arg("num_frames"),
             R"doc(
                Encode a sequence of frames from a flat int16 buffer.

                Parameters
                ----------
                depth_buffer : Sequence[int] (short)
                    Flat, contiguous int16 buffer containing all frames back-to-back.
                num_frames : int
                    Number of frames contained in `depth_buffer`.

                Returns
                -------
                List[bytes]
                    Compressed frames in order.
            )doc")
        .def("encode",
             [](VideoEncoder &self,
                py::array_t<int16_t, py::array::c_style> buf,
                int num_frames) {
                 const int ndim = buf.ndim();
                 if (ndim < 1) {
                     throw std::runtime_error("VideoEncoder.encode: buf must be at least 1D int16 C-contiguous");
                 }
                 if (num_frames <= 0) {
                     throw std::runtime_error("VideoEncoder.encode: num_frames must be > 0");
                 }
                 if (ndim >= 2) {
                     if (buf.shape(0) != num_frames) {
                         throw std::runtime_error("VideoEncoder.encode: buf.shape[0] != num_frames");
                     }
                     // Optional validation of per-frame logical size.
                     py::ssize_t per_frame = 1;
                     for (int k = 1; k < ndim; ++k) per_frame *= buf.shape(k);
                     (void)per_frame;
                 } else { // ndim == 1
                     if (buf.size() % static_cast<size_t>(num_frames) != 0) {
                         throw std::runtime_error("VideoEncoder.encode: 1D buf length not divisible by num_frames");
                     }
                 }

                 const int16_t *ptr = buf.data();
                 self.setNumFrames(num_frames);

                 std::vector<std::vector<char>> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(const_cast<short*>(reinterpret_cast<const short*>(ptr)));
                 }

                 py::list out(comp.size());
                 for (size_t i = 0; i < comp.size(); ++i) {
                     const auto &frame = comp[i];
                     out[i] = py::bytes(frame.data(), frame.size());
                 }
                 return out;
             },
             py::arg("depth_buffer_np"), py::arg("num_frames"),
             R"doc(
                Encode a sequence of frames from a NumPy array.

                Parameters
                ----------
                depth_buffer_np : numpy.ndarray[int16], C-contiguous
                    Either 1D flat (N_total,) or ND with shape (num_frames, ...).
                num_frames : int
                    Number of frames along the first dimension (or implied by 1D length).

                Returns
                -------
                List[bytes]
                    Compressed frames in order.
            )doc")
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf) {
                 // Encode using the current `num_frames` previously set.
                 std::vector<std::vector<char>> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 py::list out(comp.size());
                 for (size_t i = 0; i < comp.size(); ++i) {
                     const auto &frame = comp[i];
                     out[i] = py::bytes(frame.data(), frame.size());
                 }
                 return out;
             },
             py::arg("depth_buffer"),
             R"doc(
                Encode a sequence of frames from a flat buffer using the current num_frames.

                Parameters
                ----------
                depth_buffer : Sequence[int] (short)
                    Flat, contiguous int16 buffer containing all frames.

                Returns
                -------
                List[bytes]
                    Compressed frames in order.
            )doc")
        .def("setFrameEncoder", &VideoEncoder::setFrameEncoder,
             py::arg("encoder"),
             R"doc(Attach a different FrameEncoder to this VideoEncoder.)doc")
        .def("setFrameSize", &VideoEncoder::setFrameSize,
             py::arg("size"),
             R"doc(Set the number of int16 elements per frame.)doc")
        .def("setNumFrames", &VideoEncoder::setNumFrames,
             py::arg("n"),
             R"doc(Set the number of frames to encode/decode.)doc")
        .def("getFrameSize", &VideoEncoder::getFrameSize,
             R"doc(Return the number of int16 elements per frame.)doc")
        .def("getNumFrames", &VideoEncoder::getNumFrames,
             R"doc(Return the number of frames configured for this encoder.)doc");

    // ===== Base FrameDecoder binding =====
    py::class_<FrameDecoder>(m, "FrameDecoder", R"doc(
            Base frame decoder interface.

            Methods
            -------
            decode(compressed_frame : bytes) -> numpy.ndarray[int16]
                Decode a single compressed frame into a 1D NumPy array.
            getFrameSize() / setFrameSize(size)
                Get or set the expected elements per frame.
        )doc")
        .def("decode",
             [](FrameDecoder &self, py::bytes compressed_frame) {
                 // Zero-copy-ish input from Python bytes to raw pointer.
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p);  // C++ signature unchanged
                 // Return 1D numpy int16 (owned via capsule, without extra copies beyond cast).
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"),
             R"doc(
                Decode a single compressed frame.

                Parameters
                ----------
                compressed_frame : bytes
                    Compressed frame produced by a matching encoder.

                Returns
                -------
                numpy.ndarray[int16], shape (N,)
                    Decompressed int16 buffer.
            )doc")
        .def("getFrameSize", &FrameDecoder::getFrameSize,
             R"doc(Return the expected number of elements per frame.)doc")
        .def("setFrameSize", &FrameDecoder::setFrameSize,
             py::arg("size"),
             R"doc(Set the expected number of elements per frame.)doc");

    // ===== Base VideoDecoder binding =====
    py::class_<VideoDecoder>(m, "VideoDecoder", R"doc(
            Base video decoder interface that wraps a FrameDecoder.

            Accepts a sequence of Python `bytes` (one per frame). Decodes into either:
            - a 2D NumPy int16 array of shape (frames, per_frame) if all frames are uniform
            - a 1D NumPy int16 array concatenating all frames if frame sizes vary
        )doc")
        .def(py::init<FrameDecoder*>(), py::arg("frame_decoder"),
             R"doc(Create a VideoDecoder bound to a FrameDecoder instance.)doc")
        .def("decode",
            [](VideoDecoder &self, py::sequence frames_bytes) -> py::array {
                // Build pointer array without copying bytes; keep py::bytes alive.
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                auto frames_vec = self.decode(ptrs);

                const size_t per = frames_vec.front().size();
                const bool uniform = std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });

                if (uniform) {
                    const size_t frames = frames_vec.size();
                    std::vector<int16_t> flat(frames * per);
                    for (size_t i = 0; i < frames; ++i) {
                        std::memcpy(flat.data() + i * per,
                                    frames_vec[i].data(),
                                    per * sizeof(int16_t));
                    }
                    return numpy_from_owned_vector_2d(std::move(flat),
                                                      static_cast<ssize_t>(frames),
                                                      static_cast<ssize_t>(per));
                } else {
                    // Concatenate variable-sized frames into one flat array.
                    size_t total = 0;
                    for (const auto &v : frames_vec) total += v.size();

                    std::vector<int16_t> flat(total);
                    size_t offset = 0;
                    for (const auto &v : frames_vec) {
                        const size_t count = v.size();
                        std::memcpy(flat.data() + offset, v.data(), count * sizeof(int16_t));
                        offset += count;
                    }
                    return numpy_from_owned_vector_1d(std::move(flat));
                }
            },
            py::arg("frames_bytes"),
            R"doc(
                Decode a sequence of compressed frames.

                Parameters
                ----------
                frames_bytes : Sequence[bytes]
                    Iterable of compressed frames (one bytes object per frame).

                Returns
                -------
                numpy.ndarray[int16]
                    If frames are uniformly sized: shape (num_frames, per_frame).
                    Otherwise: shape (total_elements,), concatenation of all frames.
            )doc")
        .def("setFrameDecoder", &VideoDecoder::setFrameDecoder,
             py::arg("decoder"),
             R"doc(Attach a different FrameDecoder to this VideoDecoder.)doc")
        .def("setFrameSize", &VideoDecoder::setFrameSize,
             py::arg("size"),
             R"doc(Set the number of int16 elements per frame.)doc")
        .def("setNumFrames", &VideoDecoder::setNumFrames,
             py::arg("n"),
             R"doc(Set the number of frames expected for decoding.)doc")
        .def("getFrameSize", &VideoDecoder::getFrameSize,
             R"doc(Return the number of int16 elements per frame.)doc")
        .def("getNumFrames", &VideoDecoder::getNumFrames,
             R"doc(Return the number of frames configured for this decoder.)doc");

    // ===== TRVL Encoder (single-frame) =====
    py::class_<trvl::EncoderTRVL>(m, "EncoderTRVL", R"doc(
            TRVL single-frame encoder.

            Encodes a single frame using thresholded updates and invalidation rules tuned
            for depth images.
        )doc")
        .def(py::init<int, short, int>(),
             py::arg("frame_size"),
             py::arg("change_threshold"),
             py::arg("invalidation_threshold"),
             R"pbdoc(
                Create an Encoder.

                Parameters
                ----------
                frame_size : int
                    Number of pixels (int16 elements) per frame.
                change_threshold : int
                    Threshold for considering a pixel "changed".
                invalidation_threshold : int
                    Number of consecutive zeros after which a pixel is reset/invalidated.
            )pbdoc")
        .def("encode",
             [](trvl::EncoderTRVL &self,
                std::vector<short> &depth_buffer,
                bool keyframe) {
                 // Run native encode without the GIL for better parallelism.
                 std::vector<char> compressed;
                 {
                     py::gil_scoped_release release;
                     compressed = self.encode(static_cast<short *>(depth_buffer.data()), keyframe);
                 }
                 return py::bytes(compressed.data(), compressed.size());
             },
             py::arg("depth_buffer"),
             py::arg("keyframe") = false,
             R"doc(
                Encode a frame.

                Parameters
                ----------
                depth_buffer : Sequence[int] (short)
                    Flat, contiguous int16 buffer.
                keyframe : bool, optional
                    If True, force a keyframe.

                Returns
                -------
                bytes
                    Compressed frame.
            )doc")
        .def("encode",
            [](trvl::EncoderTRVL &self,
               py::array_t<int16_t, py::array::c_style> depth_buffer,
               bool keyframe) {
                const int ndim = depth_buffer.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "EncoderTRVL.encode: buffer must be at least 1D int16 C-contiguous");
                }
       
                const size_t elems = static_cast<size_t>(depth_buffer.size());
                const int16_t* ptr = depth_buffer.data();
       
                std::vector<char> compressed;
                {
                    py::gil_scoped_release release;  // heavy work off the GIL
                    compressed = self.encode(const_cast<short*>(
                        reinterpret_cast<const short*>(ptr)), keyframe);
                }
                return py::bytes(compressed.data(), compressed.size());
            },
            py::arg("depth_buffer_np"),
            py::arg("keyframe") = false,
            R"doc(
                Encode a frame from a NumPy array.

                Parameters
                ----------
                depth_buffer_np : numpy.ndarray[int16], C-contiguous
                    1D or ND buffer flattened internally (elements = frame_size).
                keyframe : bool, optional
                    If True, force a keyframe.

                Returns
                -------
                bytes
                    Compressed frame.
            )doc");

    // ===== TRVL Video encoder =====
    py::class_<trvl::VideoEncoderTRVL>(m, "VideoEncoderTRVL", R"doc(
            TRVL video encoder.

            Encodes multiple frames using TRVL and periodic keyframes determined by the
            keyframe interval.
        )doc")
        .def(py::init<int, int, short, int>(),
             py::arg("keyframe_interval"),
             py::arg("frame_size"),
             py::arg("change_threshold"),
             py::arg("invalidation_threshold"),
             R"doc(
                Create a TRVL video encoder.

                Parameters
                ----------
                keyframe_interval : int
                    Number of frames between keyframes.
                frame_size : int
                    Elements per frame (int16).
                change_threshold : int
                    TRVL change threshold.
                invalidation_threshold : int
                    TRVL invalidation threshold.
            )doc")
        .def("encode",
             [](trvl::VideoEncoderTRVL &self,
                std::vector<short> &depth_buffer,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 std::vector<std::vector<char>> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(depth_buffer.data());
                 }
                 py::list out(comp.size());
                 for (size_t i = 0; i < comp.size(); ++i) {
                     const auto &frame = comp[i];
                     out[i] = py::bytes(frame.data(), frame.size());
                 }
                 return out;
             },
             py::arg("depth_buffer"),
             py::arg("num_frames"),
             "Encode multiple frames with TRVL video encoder")
        .def("encode",
            [](trvl::VideoEncoderTRVL &self,
               py::array_t<int16_t, py::array::c_style> buf,
               int num_frames) {
                const int ndim = buf.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "VideoEncoderTRVL.encode: buf must be at least 1D int16 C-contiguous");
                }
                if (num_frames <= 0) {
                    throw std::runtime_error("VideoEncoderTRVL.encode: num_frames must be > 0");
                }
                if (ndim >= 2) {
                    if (buf.shape(0) != num_frames) {
                        throw std::runtime_error("VideoEncoderTRVL.encode: buf.shape[0] != num_frames");
                    }
                    py::ssize_t per_frame = 1;
                    for (int k = 1; k < ndim; ++k) per_frame *= buf.shape(k);
                    (void)per_frame;
                } else { // 1D
                    if (buf.size() % static_cast<size_t>(num_frames) != 0) {
                        throw std::runtime_error(
                            "VideoEncoderTRVL.encode: 1D buf length not divisible by num_frames");
                    }
                }

                const int16_t* ptr = buf.data();
                self.setNumFrames(num_frames);

                std::vector<std::vector<char>> comp;
                {
                    py::gil_scoped_release release;  // heavy work without the GIL
                    comp = self.encode(const_cast<short*>(
                        reinterpret_cast<const short*>(ptr)));
                }

                py::list out(comp.size());
                for (size_t i = 0; i < comp.size(); ++i) {
                    const auto &frame = comp[i];
                    out[i] = py::bytes(frame.data(), frame.size());
                }
                return out;
            },
            py::arg("buf_np"), py::arg("num_frames"),
            R"doc(
                Encode multiple frames from a NumPy buffer.

                Parameters
                ----------
                buf_np : numpy.ndarray[int16], C-contiguous
                    Either 1D flat (N_total,) or ND with shape (num_frames, ...).
                num_frames : int
                    Number of frames along axis 0 (or implied by 1D length).

                Returns
                -------
                List[bytes]
                    Compressed TRVL frames in order.
            )doc");

    // ===== TRVL Decoder =====
    py::class_<trvl::DecoderTRVL>(m, "DecoderTRVL", R"doc(
            TRVL single-frame decoder.
        )doc")
        .def(py::init<int>(),
             py::arg("frame_size"),
             R"pbdoc(
                Create a Decoder.

                Parameters
                ----------
                frame_size : int
                    Number of pixels per frame.
            )pbdoc")
        .def("decode",
             [](trvl::DecoderTRVL &self,
                py::bytes compressed_frame,
                bool keyframe) {
                 // Avoid extra copies for input. Pointer remains valid due to py::bytes lifetime.
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p, keyframe);  
                 // Return NumPy 1D int16 with owned storage.
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"),
             py::arg("keyframe") = false,
             R"pbdoc(
                Decode a single TRVL-compressed frame.

                Parameters
                ----------
                compressed_frame : bytes
                    Compressed frame produced by EncoderTRVL/VideoEncoderTRVL.
                keyframe : bool, optional
                    Whether this frame is a keyframe.

                Returns
                -------
                numpy.ndarray[int16], shape (N,)
                    Decompressed int16 buffer.
            )pbdoc");

    // ===== TRVL Video decoder =====
    py::class_<trvl::VideoDecoderTRVL>(m, "VideoDecoderTRVL", R"doc(
            TRVL video decoder.

            Decodes a sequence of frame bytes into a NumPy array. If all frames have the
            same size, returns a 2D array (frames, per_frame); otherwise returns a 1D array
            containing the concatenation of all frames.
        )doc")
        .def(py::init<int, int>(),
             py::arg("keyframe_interval"),
             py::arg("frame_size") = 0,
             R"doc(
                Create a TRVL video decoder.

                Parameters
                ----------
                keyframe_interval : int
                    Keyframe spacing used during encoding.
                frame_size : int, optional
                    Elements per frame (can be set later).
            )doc")
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::sequence frames_bytes, int frame_size) -> py::array {
                self.setFrameSize(frame_size);
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                auto frames_vec = self.decode(ptrs);  

                const size_t per = frames_vec.empty() ? 0 : frames_vec.front().size();
                const bool uniform = !frames_vec.empty() && std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });


                if (uniform) {
                    const size_t frames = frames_vec.size();
                    std::vector<int16_t> flat;
                    flat.resize(frames * per);
                    for (size_t i = 0; i < frames; ++i) {
                        std::memcpy(flat.data() + i * per,
                                    frames_vec[i].data(),
                                    per * sizeof(int16_t));
                    }
                    return numpy_from_owned_vector_2d(std::move(flat),
                                                      static_cast<ssize_t>(frames),
                                                      static_cast<ssize_t>(per));
                } else {
                    size_t total = 0;
                    for (const auto &v : frames_vec) total += v.size();

                    std::vector<int16_t> flat;
                    flat.resize(total);

                    size_t offset = 0;
                    for (const auto &v : frames_vec) {
                        const size_t bytes = v.size() * sizeof(int16_t);
                        std::memcpy(flat.data() + offset, v.data(), bytes);
                        offset += v.size();
                    }
                    return numpy_from_owned_vector_1d(std::move(flat));
                }
            }, py::arg("frames_bytes"), py::arg("frame_size"),
            R"doc(
                Decode a sequence of TRVL-compressed frames with an explicit frame size.

                Parameters
                ----------
                frames_bytes : Sequence[bytes]
                    Frame bytes in order.
                frame_size : int
                    Elements per frame. Sets decoder frame size before decoding.

                Returns
                -------
                numpy.ndarray[int16]
                    (frames, per_frame) if uniform; else (total_elements,).
            )doc")
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::sequence frames_bytes) -> py::array {
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                auto frames_vec = self.decode(ptrs); 

                const size_t per = frames_vec.empty() ? 0 : frames_vec.front().size();
                const bool uniform = !frames_vec.empty() && std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });

                if (uniform) {
                    const size_t frames = frames_vec.size();
                    std::vector<int16_t> flat;
                    flat.resize(frames * per);
                    for (size_t i = 0; i < frames; ++i) {
                        std::memcpy(flat.data() + i * per,
                                    frames_vec[i].data(),
                                    per * sizeof(int16_t));
                    }
                    return numpy_from_owned_vector_2d(std::move(flat),
                                                      static_cast<ssize_t>(frames),
                                                      static_cast<ssize_t>(per));
                } else {
                    size_t total = 0;
                    for (const auto &v : frames_vec) total += v.size();

                    std::vector<int16_t> flat;
                    flat.resize(total);

                    size_t offset = 0;
                    for (const auto &v : frames_vec) {
                        const size_t bytes = v.size() * sizeof(int16_t);
                        std::memcpy(flat.data() + offset, v.data(), bytes);
                        offset += v.size();
                    }
                    return numpy_from_owned_vector_1d(std::move(flat));
                }
            },
            py::arg("frames_bytes"),
            R"doc(
                Decode a sequence of TRVL-compressed frames.

                Parameters
                ----------
                frames_bytes : Sequence[bytes]
                    Frame bytes in order.

                Returns
                -------
                numpy.ndarray[int16]
                    (frames, per_frame) if uniform; else (total_elements,).
            )doc")
        .def("set_keyframe_interval",
             &trvl::VideoDecoderTRVL::setKeyframeInterval,
             py::arg("interval"),
             "Set the keyframe interval for the decoder");

    // ===== RVL Frame/video encoders and decoders (global namespace) =====
    py::class_<EncoderRVL>(m, "EncoderRVL", R"doc(
           RVL single-frame encoder.
        )doc")
        .def(py::init<int>(), py::arg("frame_size"),
             R"doc(Create an RVL frame encoder with the given frame size.)doc")
        .def("encode",
             [](EncoderRVL &self, std::vector<short> &buf) {
                 // Release GIL while running RVL compression.
                 std::vector<char> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"),
             R"doc(
                Encode a single frame using RVL.

                Parameters
                ----------
                depth_buffer : Sequence[int] (short)
                    Flat, contiguous int16 buffer (elements = frame_size).

                Returns
                -------
                bytes
                    Compressed frame.
            )doc")
        .def("encode",
            [](EncoderRVL &self,
               py::array_t<int16_t, py::array::c_style> depth_buffer) {
                const int ndim = depth_buffer.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "EncoderRVL.encode: buffer must be at least 1D int16 C-contiguous");
                }
       
                const size_t elems = static_cast<size_t>(depth_buffer.size());
                const int16_t* ptr = depth_buffer.data();
       
                std::vector<char> comp;
                {
                    py::gil_scoped_release release;  
                    comp = self.encode(const_cast<short*>(
                        reinterpret_cast<const short*>(ptr)));
                }
                return py::bytes(comp.data(), comp.size());
            },
            py::arg("depth_buffer_np"),
            R"doc(
                Encode a single frame from a NumPy array using RVL.

                Parameters
                ----------
                depth_buffer_np : numpy.ndarray[int16], C-contiguous
                    1D or ND buffer flattened internally.

                Returns
                -------
                bytes
                    Compressed frame.
            )doc");

    py::class_<VideoEncoderRVL>(m, "VideoEncoderRVL", R"doc(
            RVL video encoder.
        )doc")
        .def(py::init<int>(), py::arg("frame_size"),
             R"doc(Create an RVL video encoder with the given frame size.)doc")
        .def("encode",
             [](VideoEncoderRVL &self,
                std::vector<short> &buf,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 std::vector<std::vector<char>> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 py::list out(comp.size());
                 for (size_t i = 0; i < comp.size(); ++i) {
                     const auto &frame = comp[i];
                     out[i] = py::bytes(frame.data(), frame.size());
                 }
                 return out;
             },
             py::arg("buf"), py::arg("num_frames"),
             R"doc(
                Encode multiple frames using RVL from a flat buffer.

                Parameters
                ----------
                buf : Sequence[int] (short)
                    Flat, contiguous int16 buffer with all frames.
                num_frames : int
                    Number of frames in `buf`.

                Returns
                -------
                List[bytes]
                    Compressed frames in order.
            )doc")
        .def("encode",
            [](VideoEncoderRVL &self,
               py::array_t<int16_t, py::array::c_style> buf,
               int num_frames) {
                const int ndim = buf.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "VideoEncoderTRVL.encode: buf must be at least 1D int16 C-contiguous");
                }
                if (num_frames <= 0) {
                    throw std::runtime_error("VideoEncoderTRVL.encode: num_frames must be > 0");
                }
                if (ndim >= 2) {
                    if (buf.shape(0) != num_frames) {
                        throw std::runtime_error("VideoEncoderTRVL.encode: buf.shape[0] != num_frames");
                    }
                    py::ssize_t per_frame = 1;
                    for (int k = 1; k < ndim; ++k) per_frame *= buf.shape(k);
                    (void)per_frame;
                } else { // 1D
                    if (buf.size() % static_cast<size_t>(num_frames) != 0) {
                        throw std::runtime_error(
                            "VideoEncoderTRVL.encode: 1D buf length not divisible by num_frames");
                    }
                }
       
                const int16_t* ptr = buf.data();
                self.setNumFrames(num_frames);

                std::vector<std::vector<char>> comp;
                {
                    py::gil_scoped_release release;  // heavy work without the GIL
                    comp = self.encode(const_cast<short*>(
                        reinterpret_cast<const short*>(ptr)));
                }

                py::list out(comp.size());
                for (size_t i = 0; i < comp.size(); ++i) {
                    const auto &frame = comp[i];
                    out[i] = py::bytes(frame.data(), frame.size());
                }
                return out;
            },
            py::arg("buf_np"), py::arg("num_frames"),
            R"doc(
                Encode multiple frames using RVL from a NumPy array.

                Parameters
                ----------
                buf_np : numpy.ndarray[int16], C-contiguous
                    Either 1D flat (N_total,) or ND with shape (num_frames, ...).
                num_frames : int
                    Number of frames along the first dimension (or implied by 1D length).

                Returns
                -------
                List[bytes]
                    Compressed frames in order.
            )doc");
       
    py::class_<DecoderRVL>(m, "DecoderRVL", R"doc(
            RVL single-frame decoder.
        )doc")
        .def(py::init<int>(), py::arg("frame_size"),
             R"doc(Create an RVL frame decoder with the given frame size.)doc")
        .def("decode",
             [](DecoderRVL &self, py::bytes compressed_frame) {
                 // Zero-copy-ish input from Python bytes to raw pointer.
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p);  // backend signature unchanged
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"),
             R"doc(
                Decode a single RVL-compressed frame.

                Parameters
                ----------
                compressed_frame : bytes
                    Compressed frame produced by EncoderRVL/VideoEncoderRVL.

                Returns
                -------
                numpy.ndarray[int16], shape (N,)
                    Decompressed int16 buffer.
            )doc");

    // ===== RVL Video Decoder (optimized flat return when available) =====
    py::class_<VideoDecoderRVL>(m, "VideoDecoderRVL", R"doc(
            RVL video decoder.

            Uses an optimized path (`decode_flat`) to produce a single contiguous buffer and
            returns it as a 2D NumPy array of shape (num_frames, per_frame).
        )doc")
        .def(py::init<int>(), py::arg("frame_size") = 0,
             R"doc(Create an RVL video decoder. Frame size can be set later.)doc")
        .def("decode",
            [](VideoDecoderRVL &self, py::sequence frames_bytes, int frame_size) -> py::array {
                self.setFrameSize(frame_size);
                const ssize_t n = py::len(frames_bytes);
                if (n == 0) {
                    return py::array(py::dtype::of<int16_t>(), {0, 0});
                }

                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                int elems_per_frame = 0;
                std::vector<int16_t> flat = self.decode_flat(ptrs, elems_per_frame);

                const ssize_t frames = n;
                const ssize_t per = elems_per_frame;

                return numpy_from_owned_vector_2d(std::move(flat), frames, per);
            }, py::arg("frames_bytes"), py::arg("frame_size"),
            R"doc(
                Decode a sequence of RVL-compressed frames with an explicit frame size.

                Parameters
                ----------
                frames_bytes : Sequence[bytes]
                    Frame bytes in order.
                frame_size : int
                    Elements per frame. Sets decoder frame size before decoding.

                Returns
                -------
                numpy.ndarray[int16], shape (num_frames, per_frame)
                    Decompressed frames.
            )doc")
        .def("decode",
             [](VideoDecoderRVL &self, py::sequence frames_bytes) -> py::array {
                 const ssize_t n = py::len(frames_bytes);
                 if (n == 0) {
                     return py::array(py::dtype::of<int16_t>(), {0, 0});
                 }

                 std::vector<py::bytes> keepalive;
                 std::vector<char *> ptrs;
                 bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                 int elems_per_frame = 0;
                 std::vector<int16_t> flat = self.decode_flat(ptrs, elems_per_frame);

                 const ssize_t frames = n;
                 const ssize_t per = elems_per_frame;

                 return numpy_from_owned_vector_2d(std::move(flat), frames, per);
             },
             py::arg("frames_bytes"),
             R"doc(
                Decode a sequence of Python bytes objects into a 2D NumPy array (int16)
                with shape (num_frames, elements_per_frame), minimizing copies.

                Parameters
                ----------
                frames_bytes : Sequence[bytes]
                    Frame bytes in order.

                Returns
                -------
                numpy.ndarray[int16], shape (num_frames, per_frame)
                    Decompressed frames.
            )doc");
}
