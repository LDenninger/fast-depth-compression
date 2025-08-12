// bindings.cpp — optimized bindings minimizing Python<->C++ overhead
// - Zero-copy-ish bytes intake using PyBytes_AS_STRING with keepalive
// - NumPy outputs for decoders to avoid Python list-of-lists materialization
// - NumPy-first encoder overloads (C-contiguous int16) with logical flattening of trailing dims
// - GIL release around heavy native encode calls
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
                 std::vector<char> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"))
        // Efficient overload: accept NumPy (C-contiguous int16) with no intermediate copies
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
             py::arg("depth_buffer_np"))
        .def("getFrameSize", &FrameEncoder::getFrameSize)
        .def("setFrameSize", &FrameEncoder::setFrameSize);

    // ===== Base VideoEncoder binding =====
    py::class_<VideoEncoder>(m, "VideoEncoder")
        .def(py::init<FrameEncoder*>(), py::arg("frame_encoder"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf, int num_frames) {
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
             py::arg("depth_buffer"), py::arg("num_frames"))
        // Efficient overload: accept NumPy holding concatenated frames (1D, 2D, or N-D with trailing dims flattened logically)
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
                     // per-frame logical size (optional validation only)
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
             py::arg("depth_buffer_np"), py::arg("num_frames"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf) {
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
                //if (frames_vec.empty()) {
                //    return py::array(py::dtype::of<int16_t>(), {0});  // empty 1D
                //}

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

    // ===== TRVL Encoder (single-frame) =====
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
                 std::vector<char> compressed;
                 {
                     py::gil_scoped_release release;
                     compressed = self.encode(static_cast<short *>(depth_buffer.data()), keyframe);
                 }
                 return py::bytes(compressed.data(), compressed.size());
             },
             py::arg("depth_buffer"),
             py::arg("keyframe") = false)
        // Efficient overload for NumPy input (1D int16)
        .def("encode",
            [](trvl::EncoderTRVL &self,
               py::array_t<int16_t, py::array::c_style> depth_buffer,
               bool keyframe) {
                const int ndim = depth_buffer.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "EncoderTRVL.encode: buffer must be at least 1D int16 C-contiguous");
                }
       
                // Logical flattening of all dimensions (no reshape/copy needed).
                const size_t elems = static_cast<size_t>(depth_buffer.size());
       
                // Optional safety: ensure total elements match the encoder's frame size.
                // Uncomment if you want strict validation.
                // if (elems != static_cast<size_t>(self.getFrameSize())) {
                //     throw std::runtime_error("EncoderTRVL.encode: flattened element count "
                //                              "does not match encoder frame_size");
                // }
       
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
        // Efficient overload for NumPy input (1D, 2D, or N-D; flatten trailing dims logically)
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
            py::arg("buf_np"), py::arg("num_frames"));

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
             py::arg("frame_size") = 0)
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::sequence frames_bytes, int frame_size) -> py::array {
                self.setFrameSize(frame_size);
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                auto frames_vec = self.decode(ptrs);  // backend signature unchanged

                const size_t per = frames_vec.empty() ? 0 : frames_vec.front().size();
                const bool uniform = !frames_vec.empty() && std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });

                //if (frames_vec.empty()) {
                //    // Return an empty 1D int16 array
                //    return py::array(py::dtype::of<int16_t>(), {0});
                //}

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
            }, py::arg("frames_bytes"), py::arg("frame_size"))
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::sequence frames_bytes) -> py::array {
                std::vector<py::bytes> keepalive;
                std::vector<char *> ptrs;
                bytes_sequence_to_ptrs_with_keepalive(frames_bytes, keepalive, ptrs);

                auto frames_vec = self.decode(ptrs);  // backend signature unchanged

                const size_t per = frames_vec.empty() ? 0 : frames_vec.front().size();
                const bool uniform = !frames_vec.empty() && std::all_of(
                    frames_vec.begin(), frames_vec.end(),
                    [per](const std::vector<short> &v) { return v.size() == per; });

                //if (frames_vec.empty()) {
                //    // Return an empty 1D int16 array
                //    return py::array(py::dtype::of<int16_t>(), {0});
                //}

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
                 std::vector<char> comp;
                 {
                     py::gil_scoped_release release;
                     comp = self.encode(buf.data());
                 }
                 return py::bytes(comp.data(), comp.size());
             },
             py::arg("depth_buffer"))
        // Efficient overload: NumPy input (1D int16)
        .def("encode",
            [](EncoderRVL &self,
               py::array_t<int16_t, py::array::c_style> depth_buffer) {
                const int ndim = depth_buffer.ndim();
                if (ndim < 1) {
                    throw std::runtime_error(
                        "EncoderRVL.encode: buffer must be at least 1D int16 C-contiguous");
                }
       
                // Logical flattening of all dimensions (no reshape/copy needed).
                const size_t elems = static_cast<size_t>(depth_buffer.size());
       
                // Optional safety check: ensure total elements match configured frame size.
                // if (elems != static_cast<size_t>(self.getFrameSize())) {
                //     throw std::runtime_error("EncoderRVL.encode: flattened element count "
                //                              "does not match encoder frame_size");
                // }
       
                const int16_t* ptr = depth_buffer.data();
       
                std::vector<char> comp;
                {
                    py::gil_scoped_release release;  // run the heavy C++ without the GIL
                    comp = self.encode(const_cast<short*>(
                        reinterpret_cast<const short*>(ptr)));
                }
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
             py::arg("buf"), py::arg("num_frames"))
        // Efficient overload: NumPy input (1D, 2D, or N-D; flatten trailing dims logically)
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
        .def(py::init<int>(), py::arg("frame_size") = 0)
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
            }, py::arg("frames_bytes"), py::arg("frame_size"))
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
             )doc");
}

/*
Notes:

1) Encoder paths:
   - NumPy overloads require C-contiguous int16 (`py::array_t<int16_t, py::array::c_style>`).
     This avoids hidden copies from `forcecast`/`mutable_data()`.
   - For N-D inputs (>= 2D), frames are axis-0; trailing dims are logically flattened.
   - `py::gil_scoped_release` surrounds native `encode` calls.

2) Vector overloads are kept for compatibility, but they will copy if Python lists are passed.
   Prefer the NumPy overloads for performance.

3) Decoder paths already avoid per-element Python overhead by returning NumPy arrays.

4) If your RVL VideoDecoder lacks `decode_flat`, replace that call with `self.decode(ptrs)` and
   flatten like the generic VideoDecoder block above.
*/
