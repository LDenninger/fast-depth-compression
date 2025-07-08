// bindings/bindings.cpp

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // for automatic std::vector ↔ Python list conversion
#include "rvl.h"
#include "trvl.h"                // your header above

namespace py = pybind11;

PYBIND11_MODULE(fdc_bindings, m) {
    m.doc() = "Python bindings for the fdcomp C++ library";

    // Encoder
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
                const std::vector<short> &depth_buffer,
                bool keyframe) {
                 // call into C++
                 auto compressed = self.encode(
                   const_cast<short*>(depth_buffer.data()),
                   keyframe);
                 // return as Python bytes
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

    // Decoder
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
                 // convert Python bytes → std::vector<char>
                 std::string tmp = compressed_frame;
                 std::vector<char> buf(tmp.begin(), tmp.end());
                 // call into C++
                 auto decoded = self.decode(buf.data(), keyframe);
                 // return as Python list[int]
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

    m.def("CompressRVL",
        [] (const std::vector<short> &input_buffer) {
            int num_pixels = static_cast<int>(input_buffer.size());
            // pre-allocate output; worst-case same size as input
            std::vector<char> output(num_pixels);
            int bytes_written = wilson::CompressRVL(
                const_cast<short*>(input_buffer.data()),
                output.data(),
                num_pixels
            );
            // shrink to actual size
            output.resize(bytes_written);
            return py::bytes(output.data(), output.size());
        },
        py::arg("depth_buffer"),
        "Compress an array of 16-bit depth values using the RVL algorithm.\n\n"
        "Returns\n"
        "-------\n"
        "bytes\n"
        "    The compressed bitstream."
    );

    m.def("DecompressRVL",
        [] (py::bytes compressed, int num_pixels) {
            // convert Python bytes → vector<char>
            std::string tmp = compressed;
            std::vector<char> input(tmp.begin(), tmp.end());
            // allocate output
            std::vector<short> output(num_pixels);
            wilson::DecompressRVL(
                input.data(),
                output.data(),
                num_pixels
            );
            return output;  // auto-converted to List[int]
        },
        py::arg("compressed_data"),
        py::arg("num_pixels"),
        "Decompress an RVL-compressed bitstream into a list of 16-bit depth values.\n\n"
        "Parameters\n"
        "----------\n"
        "compressed_data : bytes\n"
        "    The RVL-compressed bitstream.\n"
        "num_pixels : int\n"
        "    Number of depth values expected in output.\n\n"
        "Returns\n"
        "-------\n"
        "List[int]\n"
        "    The decompressed depth buffer."
    );

    m.def("RVLCompress",
        [] (const std::vector<short> &input_buffer) {
            int num_pixels = static_cast<int>(input_buffer.size());
            // call rvl::compress
            auto compressed = rvl::compress(
                const_cast<short*>(input_buffer.data()),
                num_pixels
            );
            // return as Python bytes
            return py::bytes(compressed.data(), compressed.size());
        },
        py::arg("depth_buffer"),
        "Compress an array of 16-bit depth values using the RVL algorithm (rvl namespace version).\n\n"
        "Parameters\n"
        "----------\n"
        "depth_buffer : List[int]\n"
        "    Array of 16-bit depth values to compress.\n\n"
        "Returns\n"
        "-------\n"
        "bytes\n"
        "    The compressed bitstream."
    );

    m.def("RVLDecompress",
        [] (py::bytes compressed, int num_pixels) {
            // convert Python bytes → vector<char>
            std::string tmp = compressed;
            std::vector<char> input(tmp.begin(), tmp.end());
            // call rvl::decompress
            auto output = rvl::decompress(input.data(), num_pixels);
            return output;  // auto-converted to List[int]
        },
        py::arg("compressed_data"),
        py::arg("num_pixels"),
        "Decompress an RVL-compressed bitstream into a list of 16-bit depth values (rvl namespace version).\n\n"
        "Parameters\n"
        "----------\n"
        "compressed_data : bytes\n"
        "    The RVL-compressed bitstream.\n"
        "num_pixels : int\n"
        "    Number of depth values expected in output.\n\n"
        "Returns\n"
        "-------\n"
        "List[int]\n"
        "    The decompressed depth buffer."
    );
}
