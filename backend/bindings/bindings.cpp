// bindings.cpp — optimized bindings minimizing Python<->C++ overhead
// - Zero-copy-ish bytes intake using PyBytes_AS_STRING with keepalive
// - NumPy outputs for decoders to avoid Python list-of-lists materialization
// - Extra overloads for encoders to accept NumPy buffers directly (pointer pass-through)
// - C++ backend method signatures are unchanged; only binding glue is optimized

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

static inline py::array numpy_from_owned_vector_1d(std::vector<int16_t> &&vec) {
    auto *vec_ptr = new std::vector<int16_t>(std::move(vec));
    int16_t *data_ptr = vec_ptr->data();
    const ssize_t n = static_cast<ssize_t>(vec_ptr->size());

    py::capsule free_when_done(vec_ptr, [](void *p) {
        delete static_cast<std::vector<int16_t> *>(p);
    });

    return py::array(
        py::buffer_info(
            data_ptr,
            sizeof(int16_t),
            py::format_descriptor<int16_t>::format(),
            1,
            std::vector<ssize_t>{n},
            std::vector<ssize_t>{static_cast<ssize_t>(sizeof(int16_t))}),
        free_when_done);
}

static inline py::array numpy_from_owned_vector_2d(std::vector<int16_t> &&vec,
                                                   ssize_t rows,
                                                   ssize_t cols) {
    auto *vec_ptr = new std::vector<int16_t>(std::move(vec));
    int16_t *data_ptr = vec_ptr->data();

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
            std::vector<ssize_t>{stride0, stride1}),
        free_when_done);
}

template <typename Seq>
static inline void bytes_sequence_to_ptrs_with_keepalive(const Seq &frames_bytes,
                                                         std::vector<py::bytes> &keepalive,
                                                         std::vector<char *> &ptrs) {
    const ssize_t n = py::len(frames_bytes);
    keepalive.reserve(static_cast<size_t>(n));
    ptrs.reserve(static_cast<size_t>(n));
    for (py::handle h : frames_bytes) {
        py::bytes b = py::cast<py::bytes>(h);  // type-check & incref
        keepalive.emplace_back(b);
        char *p = const_cast<char *>(PyBytes_AS_STRING(b.ptr()));
        ptrs.push_back(p);
    }
}

// ------------------------ Module ------------------------

PYBIND11_MODULE(fdc_bindings, m) {
    m.doc() = "Python bindings for the fdcomp C++ library (optimized)";

    // ===== Base FrameEncoder binding =====
    py::class_<FrameEncoder>(m, "FrameEncoder")
        // abstract, no constructor
        .def("encode",
             [](FrameEncoder &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"))
        // Efficient overload: accept NumPy (C-contiguous) without intermediate std::vector copies
        .def("encode",
             [](FrameEncoder &self, py::array_t<int16_t, py::array::c_style | py::array::forcecast> arr) {
                 if (arr.ndim() != 1) {
                     throw std::runtime_error("FrameEncoder.encode expects a 1D int16 array for a single frame.");
                 }
                 auto comp = self.encode(reinterpret_cast<short *>(arr.mutable_data()));
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer_np"))
        .def("getFrameSize", &FrameEncoder::getFrameSize)
        .def("setFrameSize", &FrameEncoder::setFrameSize);

    // ===== Base VideoEncoder binding =====
    py::class_<VideoEncoder>(m, "VideoEncoder")
        .def(py::init<FrameEncoder*>(), py::arg("frame_encoder"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf, int num_frames) {
                 self.setNumFrames(num_frames);
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer"), py::arg("num_frames"))
        // Efficient overload: accept NumPy holding concatenated frames
        .def("encode",
             [](VideoEncoder &self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> arr,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 if (arr.ndim() == 2) {
                     // (frames, elems_per_frame) -> flatten view for pointer
                     if (arr.shape(0) != num_frames) {
                         throw std::runtime_error("VideoEncoder.encode: arr.shape[0] != num_frames");
                     }
                 }
                 auto comp = self.encode(reinterpret_cast<short *>(arr.mutable_data()));
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer_np"), py::arg("num_frames"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer"))
        .def("setFrameEncoder", &VideoEncoder::setFrameEncoder)
        .def("setFrameSize", &VideoEncoder::setFrameSize)
        .def("setNumFrames", &VideoEncoder::setNumFrames)
        .def("getFrameSize", &VideoEncoder::getFrameSize)
        .def("getNumFrames", &VideoEncoder::getNumFrames);

    // ===== Base FrameDecoder binding =====
    py::class_<FrameDecoder>(m, "FrameDecoder")
        // abstract, no constructor
        .def("decode",
             [](FrameDecoder &self, py::bytes compressed_frame) {
                 // Avoid intermediate std::string copy
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p);  // C++ signature unchanged
                 // Return NumPy 1D int16 to minimize Python overhead
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"))
        .def("getFrameSize", &FrameDecoder::getFrameSize)
        .def("setFrameSize", &FrameDecoder::setFrameSize);

    // ===== Base VideoDecoder binding =====
    py::class_<VideoDecoder>(m, "VideoDecoder")
        .def(py::init<FrameDecoder*>(), py::arg("frame_decoder"))
        .def("decode",
            [](VideoDecoder &self, py::sequence frames_bytes) -> py::array {
                // Build pointer array without copying bytes
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);
       
                // Backend returns std::vector<std::vector<short>>; keep signature the same
                auto frames_vec = self.decode(ptrs);
       
                // Always return a NumPy array:
                // - uniform frame sizes  -> 2D array (frames, elems_per_frame)
                // - non-uniform sizes    -> concatenated 1D array
                if (frames_vec.empty()) {
                    return py::array(py::dtype::of<int16_t>(), {0});  // empty 1D
                }
       
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
                    // Concatenate all frames into one large 1D array
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
            py::arg("frames_bytes"))
       
        .def("setFrameDecoder", &VideoDecoder::setFrameDecoder)
        .def("setFrameSize", &VideoDecoder::setFrameSize)
        .def("setNumFrames", &VideoDecoder::setNumFrames)
        .def("getFrameSize", &VideoDecoder::getFrameSize)
        .def("getNumFrames", &VideoDecoder::getNumFrames);

    // ===== TRVL Encoder =====
    py::class_<trvl::EncoderTRVL>(m, "EncoderTRVL")
        .def(py::init<int, short, int>(),
             py::arg("frame_size"),
             py::arg("change_threshold"),
             py::arg("invalidation_threshold"),
             R"pbdoc(
             Create an Encoder.

             Parameters:
               frame_size: number of pixels per frame
               change_threshold: threshold for updating pixel
               invalidation_threshold: how many zeros to reset pixel
             )pbdoc")
        .def("encode",
             [](trvl::EncoderTRVL &self,
                std::vector<short> &depth_buffer,
                bool keyframe) {
                 auto compressed = self.encode(static_cast<short *>(depth_buffer.data()), keyframe);
                 return py::bytes(compressed.data(), compressed.size());
             },
             py::arg("depth_buffer"),
             py::arg("keyframe") = false)
        // Efficient overload for NumPy input
        .def("encode",
             [](trvl::EncoderTRVL &self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> depth_buffer,
                bool keyframe) {
                 if (depth_buffer.ndim() != 1) {
                     throw std::runtime_error("EncoderTRVL.encode expects a 1D int16 array for a single frame.");
                 }
                 auto compressed = self.encode(
                     reinterpret_cast<short *>(depth_buffer.mutable_data()),
                     keyframe);
                 return py::bytes(compressed.data(), compressed.size());
             },
             py::arg("depth_buffer_np"),
             py::arg("keyframe") = false);

    // ===== TRVL Video encoder =====
    py::class_<trvl::VideoEncoderTRVL>(m, "VideoEncoderTRVL")
        .def(py::init<int, int, short, int>(),
             py::arg("keyframe_interval"),
             py::arg("frame_size"),
             py::arg("change_threshold"),
             py::arg("invalidation_threshold"))
        .def("encode",
             [](trvl::VideoEncoderTRVL &self,
                std::vector<short> &depth_buffer,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 auto comp = self.encode(depth_buffer.data());
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer"),
             py::arg("num_frames"),
             "Encode multiple frames with TRVL video encoder")
        // Efficient overload for NumPy input
        .def("encode",
             [](trvl::VideoEncoderTRVL &self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> depth_buffer,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 if (depth_buffer.ndim() == 2 && depth_buffer.shape(0) != num_frames) {
                     throw std::runtime_error("VideoEncoderTRVL.encode: arr.shape[0] != num_frames");
                 }
                 auto comp = self.encode(reinterpret_cast<short *>(depth_buffer.mutable_data()));
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer_np"),
             py::arg("num_frames"));

    // ===== TRVL Decoder =====
    py::class_<trvl::DecoderTRVL>(m, "DecoderTRVL")
        .def(py::init<int>(),
             py::arg("frame_size"),
             R"pbdoc(
             Create a Decoder.

             Parameters:
               frame_size: number of pixels per frame
             )pbdoc")
        .def("decode",
             [](trvl::DecoderTRVL &self,
                py::bytes compressed_frame,
                bool keyframe) {
                 // Avoid extra copies for input
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p, keyframe);  // backend signature unchanged
                 // Return NumPy 1D int16
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"),
             py::arg("keyframe") = false,
             R"pbdoc(
             Decode a frame.

             Returns:
               numpy.ndarray[int16]: the decompressed depth buffer (1D)
             )pbdoc");

    // ===== TRVL Video decoder =====
    py::class_<trvl::VideoDecoderTRVL>(m, "VideoDecoderTRVL")
        .def(py::init<int, int>(),
             py::arg("keyframe_interval"),
             py::arg("frame_size"))
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::sequence frames_bytes) -> py::array {
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);
        
                auto frames_vec = self.decode(ptrs);  // backend signature unchanged
        
                if (frames_vec.empty()) {
                    // Return an empty 1D int16 array
                    return py::array(py::dtype::of<int16_t>(), {0});
                }
        
                const size_t per = frames_vec.front().size();
                const bool uniform = std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });
        
                if (uniform) {
                    // Pack into a single 2D NumPy array (int16)
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
                    // Fallback: concatenate all frames into one large 1D array (int16)
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
            py::arg("frames_bytes"))
        .def("set_keyframe_interval",
             &trvl::VideoDecoderTRVL::setKeyframeInterval,
             py::arg("interval"),
             "Set the keyframe interval for the decoder");

    // ===== RVL Frame/video encoders and decoders (global namespace) =====
    py::class_<EncoderRVL>(m, "EncoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("encode",
             [](EncoderRVL &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"))
        // Efficient overload: NumPy input
        .def("encode",
             [](EncoderRVL &self, py::array_t<int16_t, py::array::c_style | py::array::forcecast> arr) {
                 if (arr.ndim() != 1) {
                     throw std::runtime_error("EncoderRVL.encode expects a 1D int16 array for a single frame.");
                 }
                 auto comp = self.encode(reinterpret_cast<short *>(arr.mutable_data()));
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer_np"));

    py::class_<VideoEncoderRVL>(m, "VideoEncoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("encode",
             [](VideoEncoderRVL &self,
                std::vector<short> &buf,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("buf"), py::arg("num_frames"))
        // Efficient overload: NumPy input
        .def("encode",
             [](VideoEncoderRVL &self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> buf,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 if (buf.ndim() == 2 && buf.shape(0) != num_frames) {
                     throw std::runtime_error("VideoEncoderRVL.encode: arr.shape[0] != num_frames");
                 }
                 auto comp = self.encode(reinterpret_cast<short *>(buf.mutable_data()));
                 py::list py_frames;
                 //py_frames.reserve(comp.size());
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("buf_np"), py::arg("num_frames"));

    py::class_<DecoderRVL>(m, "DecoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("decode",
             [](DecoderRVL &self, py::bytes compressed_frame) {
                 // Zero-copy-ish input
                 char *p = const_cast<char *>(PyBytes_AS_STRING(compressed_frame.ptr()));
                 std::vector<short> decoded = self.decode(p);  // backend signature unchanged
                 std::vector<int16_t> out(decoded.begin(), decoded.end());
                 return numpy_from_owned_vector_1d(std::move(out));
             },
             py::arg("compressed_frame"));

    // ===== RVL Video Decoder (optimized flat return when available) =====
    py::class_<VideoDecoderRVL>(m, "VideoDecoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("decode",
             [](VideoDecoderRVL &self, py::sequence frames_bytes) -> py::array {
                 const ssize_t n = py::len(frames_bytes);
                 if (n == 0) {
                     return py::array(py::dtype::of<int16_t>(), {0, 0});
                 }

                 std::vector<py::bytes> keepalive;
                 std::vector<char *> ptrs;
                 bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                 // Prefer the contiguous backend if your C++ class implements it (signature unchanged for backend calls)
                 int elems_per_frame = 0;
                 std::vector<int16_t> flat = self.decode_flat(ptrs, elems_per_frame);
                 // If decode_flat is not available in your C++ build, replace the line above with:
                 // auto frames_vec = self.decode(ptrs);  // and then flatten here like the generic VideoDecoder path

                 const ssize_t frames = n;
                 const ssize_t per = elems_per_frame;

                 return numpy_from_owned_vector_2d(std::move(flat), frames, per);
             },
             py::arg("frames_bytes"),
             R"doc(
                 Decode a sequence of Python bytes objects into a 2D NumPy array (int16)
                 with shape (num_frames, elements_per_frame), minimizing copies.
             )doc");
}

/*
Notes:

1) Backend signatures:
   - All calls into your C++ classes (encode/decode that take pointers) are unchanged.
   - We added binding overloads that accept NumPy arrays on the encoder side for efficiency.
     Your existing vector-based overloads remain, so external API calls won't break.

2) Output types:
   - Decoders now return NumPy arrays (int16) instead of Python lists to eliminate per-element boxing.
     If you must retain the exact Python-level return types (lists), this would reintroduce overhead.
     Consider updating downstream code to consume NumPy arrays for best performance.

3) Video decoders with variable frame sizes:
   - If frames are not uniform in length, we return a Python list of 1D NumPy arrays
     (still much faster than list-of-Python-ints).

4) If your build of VideoDecoderRVL does NOT provide `decode_flat`, replace its use with
   `self.decode(ptrs)` and flatten in the binding (same as the generic VideoDecoder path).
*/
