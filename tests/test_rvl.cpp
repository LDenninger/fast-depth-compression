#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>             
#include <opencv2/opencv.hpp>   
#include "../backend/cpp/include/rvl.h"

std::vector<int> array_shape = {8, 704, 1280};

// Load flattened depth data from text file, comma-separated floats
bool load_flat_data(const std::string& filename, std::vector<short>& out) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            float f = std::stof(token);
            out.push_back(static_cast<short>(std::lround(f)));
        } catch (...) {
            // skip invalid entries
        }
    }
    return true;
}

// Save a single depth frame (short*) to two PNGs using OpenCV
// - 16-bit (scaled to 0..65535): <base>_u16.png
// - 8-bit (scaled to 0..255):    <base>_u8.png
void save_depth_frame_visuals(const short* data, int width, int height, const std::string& base_path) {
    if (!data) return;

    short vmin = data[0], vmax = data[0];
    const int N = width * height;
    for (int i = 1; i < N; ++i) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }

    const bool constant = (vmax == vmin);
    const double range = constant ? 1.0 : static_cast<double>(vmax - vmin);

    cv::Mat img8(height, width, CV_8UC1);

    for (int i = 0; i < N; ++i) {
        double norm = (static_cast<double>(data[i]) - static_cast<double>(vmin)) / range;
        if (norm < 0.0) norm = 0.0;
        if (norm > 1.0) norm = 1.0;
        uint8_t  v8  = static_cast<uint8_t>(std::lround(norm * 255.0));
        img8.at<uint8_t>(i / width, i % width) = v8;
    }

    cv::imwrite(base_path + ".png", img8);
}

int main(int argc, char** argv) {
    std::string path = (argc > 1 ? argv[1] : std::string("examples/dflat.txt"));
    std::vector<short> depth;
    if (!load_flat_data(path, depth)) return 1;

    const int n_frames = 8;
    const int height = 704;
    const int width = 1280;
    const int frame_pixels = height * width;

    if ((int)depth.size() != n_frames * frame_pixels) {
        std::cerr << "Data size mismatch: expected " << n_frames * frame_pixels
                  << " elements, got " << depth.size() << std::endl;
        return 1;
    }

    // Frame-level RVL test
    {
        EncoderRVL encoder(frame_pixels);
        DecoderRVL decoder(frame_pixels);
        short* first_frame = depth.data();

        save_depth_frame_visuals(first_frame, width, height, "frame_before");

        auto start_fenc = std::chrono::high_resolution_clock::now();
        auto compressed = encoder.encode(first_frame);
        auto end_fenc   = std::chrono::high_resolution_clock::now();
        std::cout << "Frame encode time: "
                  << std::chrono::duration<double, std::milli>(end_fenc - start_fenc).count()
                  << " ms\n";

        auto start_fdec    = std::chrono::high_resolution_clock::now();
        auto decompressed  = decoder.decode(compressed.data());
        auto end_fdec      = std::chrono::high_resolution_clock::now();
        std::cout << "Frame decode time: "
                  << std::chrono::duration<double, std::milli>(end_fdec - start_fdec).count()
                  << " ms\n";

        if ((int)decompressed.size() != frame_pixels) {
            std::cerr << "Decoded size mismatch\n";
            return 1;
        }
        for (int i = 0; i < frame_pixels; ++i) {
            if (decompressed[i] != first_frame[i]) {
                std::cerr << "Value mismatch at index " << i << "\n";
                return 1;
            }
        }

        save_depth_frame_visuals(decompressed.data(), width, height, "frame_after");

        std::cout << "RVL frame encoding/decoding test passed.\n";
    }

    // Video-level RVL test (now using VideoEncoderRVL like Python tests)
    {
        VideoEncoderRVL venc(frame_pixels);
        VideoDecoderRVL vdec(frame_pixels);

        auto start_venc = std::chrono::high_resolution_clock::now();
        auto compressed_video = venc.encode(depth.data(), n_frames);
        auto end_venc   = std::chrono::high_resolution_clock::now();
        std::cout << "Video encode time: "
                  << std::chrono::duration<double, std::milli>(end_venc - start_venc).count()
                  << " ms\n";

        if ((int)compressed_video.size() != n_frames) {
            std::cerr << "Compressed video frame count mismatch\n";
            return 1;
        }

        for (int f = 0; f < n_frames; ++f) {
            short* frame_ptr = depth.data() + f * frame_pixels;
            save_depth_frame_visuals(frame_ptr, width, height, "video_before_f" + std::to_string(f));
        }

        vdec.setNumFrames(n_frames);
        std::vector<char*> compr_video;
        for (auto& buf : compressed_video) {
            compr_video.push_back(buf.data());
        }

        auto start_vdec = std::chrono::high_resolution_clock::now();
        auto decoded_frames = vdec.decode(compr_video);
        auto end_vdec   = std::chrono::high_resolution_clock::now();
        std::cout << "Video decode time: "
                  << std::chrono::duration<double, std::milli>(end_vdec - start_vdec).count()
                  << " ms\n";

        if ((int)decoded_frames.size() != n_frames) {
            std::cerr << "Decoded video frame count mismatch\n";
            return 1;
        }

        for (int f = 0; f < n_frames; ++f) {
            const auto &decoded = decoded_frames[f];
            if ((int)decoded.size() != frame_pixels) {
                std::cerr << "Decoded frame size mismatch at frame " << f << "\n";
                return 1;
            }
            const short* original = depth.data() + f * frame_pixels;
            for (int i = 0; i < frame_pixels; ++i) {
                if (decoded[i] != original[i]) {
                    std::cerr << "Video mismatch at frame " << f << ", idx " << i << "\n";
                    return 1;
                }
            }
            save_depth_frame_visuals(decoded.data(), width, height, "video_after_f" + std::to_string(f));
        }

        std::cout << "RVL video encoding/decoding test passed.\n";
    }

    return 0;
}
