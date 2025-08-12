#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // for automatic std::vector ↔ Python list conversion
#include <string>
#include <vector>
#include <algorithm>

#include "rvl.h"
#include "trvl.h"

namespace py = pybind11;

PYBIND11_MODULE(fdc_bindings, m) {
    m.doc() = "Python bindings for the fdcomp C++ library";

    // Base FrameEncoder binding
    py::class_<FrameEncoder>(m, "FrameEncoder")
        // abstract, no constructor
        .def("encode",
             [](FrameEncoder &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 return py::bytes(comp.data(), comp.size());
             }
             ,
             py::arg("depth_buffer"))
        .def("getFrameSize", &FrameEncoder::getFrameSize)
        .def("setFrameSize", &FrameEncoder::setFrameSize);

    // Base VideoEncoder binding
    py::class_<VideoEncoder>(m, "VideoEncoder")
        .def(py::init<FrameEncoder*>(), py::arg("frame_encoder"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf, int num_frames) {
                 self.setNumFrames(num_frames);
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer"), py::arg("num_frames"))
        .def("encode",
             [](VideoEncoder &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
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

    // Base FrameDecoder binding
    py::class_<FrameDecoder>(m, "FrameDecoder")
        // abstract, no constructor
        .def("decode",
             [](FrameDecoder &self, py::bytes b) {
                 std::string s = b; // keep storage alive
                 // base expects char*, so const_cast if needed
                 return self.decode(const_cast<char*>(s.data()));
             },
             py::arg("compressed_frame"))
        .def("getFrameSize", &FrameDecoder::getFrameSize)
        .def("setFrameSize", &FrameDecoder::setFrameSize);

    // Base VideoDecoder binding
    py::class_<VideoDecoder>(m, "VideoDecoder")
        .def(py::init<FrameDecoder*>(), py::arg("frame_decoder"))
        .def("decode",
            [](VideoDecoder &self, py::list frames_bytes) {
                const auto n = py::len(frames_bytes);
   
                std::vector<std::string> storage;
                storage.reserve(n);
   
                std::vector<char*> ptrs;
                ptrs.reserve(n);
   
                for (py::handle h : frames_bytes) {
                    // Ensure item is bytes and copy into std::string
                    py::bytes pb = py::cast<py::bytes>(h);            // Type-checks & converts
                    storage.emplace_back(static_cast<std::string>(pb)); // copy bytes -> string
   
                    // Pointer to the underlying data in storage (lifetime is OK)
                    char* p = storage.back().empty()
                            ? nullptr
                            : const_cast<char*>(storage.back().data()); // use &storage.back()[0] if you prefer
                    ptrs.push_back(p);
                }
   
                return self.decode(ptrs);
            }, py::arg("frames_bytes"))
        .def("setFrameDecoder", &VideoDecoder::setFrameDecoder)
        .def("setFrameSize", &VideoDecoder::setFrameSize)
        .def("setNumFrames", &VideoDecoder::setNumFrames)
        .def("getFrameSize", &VideoDecoder::getFrameSize)
        .def("getNumFrames", &VideoDecoder::getNumFrames);

    // TRVL Encoder
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
                 auto compressed = self.encode(
                   static_cast<short*>(depth_buffer.data()),
                   keyframe);
                 return py::bytes(compressed.data(), compressed.size());
             },
             py::arg("depth_buffer"),
             py::arg("keyframe") = false,
             R"pbdoc(
             Encode a frame.

             Parameters:
               depth_buffer: sequence of ints (short) length==frame_size
               keyframe: if true, do full-frame encoding
             Returns:
               bytes: the compressed frame
             )pbdoc");

    // TRVL Video encoder
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
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
                 return py_frames;
             },
             py::arg("depth_buffer"),
             py::arg("num_frames"),
             "Encode multiple frames with TRVL video encoder");

    // TRVL Decoder
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
                 std::string tmp = compressed_frame;
                 std::vector<char> buf(tmp.begin(), tmp.end());
                 auto decoded = self.decode(buf.data(), keyframe);
                 return decoded;
             },
             py::arg("compressed_frame"),
             py::arg("keyframe") = false,
             R"pbdoc(
             Decode a frame.

             Parameters:
               compressed_frame: bytes from Encoder.encode
               keyframe: if true, treat as keyframe
             Returns:
               List[int]: the decompressed depth buffer
             )pbdoc");

    // TRVL Video decoder
    py::class_<trvl::VideoDecoderTRVL>(m, "VideoDecoderTRVL")
        .def(py::init<int, int>(),
             py::arg("keyframe_interval"),
             py::arg("frame_size"))
        .def("decode",
            [](trvl::VideoDecoderTRVL &self, py::list frames_bytes) {
                const auto n = py::len(frames_bytes);
   
                std::vector<std::string> storage;
                storage.reserve(n);
   
                std::vector<char*> ptrs;
                ptrs.reserve(n);
   
                for (py::handle h : frames_bytes) {
                    // Ensure item is bytes and copy into std::string
                    py::bytes pb = py::cast<py::bytes>(h);            // Type-checks & converts
                    storage.emplace_back(static_cast<std::string>(pb)); // copy bytes -> string
   
                    // Pointer to the underlying data in storage (lifetime is OK)
                    char* p = storage.back().empty()
                            ? nullptr
                            : const_cast<char*>(storage.back().data()); // use &storage.back()[0] if you prefer
                    ptrs.push_back(p);
                }
   
                return self.decode(ptrs);
            },
            py::arg("frames_bytes"))
        .def("set_keyframe_interval",
             &trvl::VideoDecoderTRVL::setKeyframeInterval,
             py::arg("interval"),
             "Set the keyframe interval for the decoder");

    // RVL Frame/video encoders and decoders (global namespace)
    py::class_<EncoderRVL>(m, "EncoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("encode",
             [](EncoderRVL &self, std::vector<short> &buf) {
                 auto comp = self.encode(buf.data());
                 return py::bytes(comp.data(), comp.size());
             }, py::arg("depth_buffer"));

    py::class_<VideoEncoderRVL>(m, "VideoEncoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("encode",
             [](VideoEncoderRVL &self,
                std::vector<short> &buf,
                int num_frames) {
                 self.setNumFrames(num_frames);
                 auto comp = self.encode(buf.data());
                 py::list py_frames;
                 for (auto &frame : comp) {
                     py_frames.append(py::bytes(frame.data(), frame.size()));
                 }
        
                 return py_frames;
             }, py::arg("buf"), py::arg("num_frames"));

    py::class_<DecoderRVL>(m, "DecoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("decode",
             [](DecoderRVL &self, py::bytes compressed_frame) {
                 std::string s = compressed_frame;
                 return self.decode(const_cast<char*>(s.data()));
             }, py::arg("compressed_frame"));


    // RVL Video Decoder
    py::class_<VideoDecoderRVL>(m, "VideoDecoderRVL")
        .def(py::init<int>(), py::arg("frame_size"))
        .def("decode",
            [](VideoDecoderRVL &self, py::list frames_bytes) {
                const auto n = py::len(frames_bytes);
   
                std::vector<std::string> storage;
                storage.reserve(n);
   
                std::vector<char*> ptrs;
                ptrs.reserve(n);
   
                for (py::handle h : frames_bytes) {
                    // Ensure item is bytes and copy into std::string
                    py::bytes pb = py::cast<py::bytes>(h);            // Type-checks & converts
                    storage.emplace_back(static_cast<std::string>(pb)); // copy bytes -> string
   
                    // Pointer to the underlying data in storage (lifetime is OK)
                    char* p = storage.back().empty()
                            ? nullptr
                            : const_cast<char*>(storage.back().data()); // use &storage.back()[0] if you prefer
                    ptrs.push_back(p);
                }
   
                return self.decode(ptrs);
            },
            py::arg("frames_bytes"));
}
