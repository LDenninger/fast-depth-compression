#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp> // Added for frame visualization
#include <chrono>             // added for timing
#include "../backend/cpp/include/trvl.h"

std::vector<int> array_shape = {8, 704, 1280}; // For reference (matches test_rvl.cpp)

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

// Save a single depth frame (short*) to 8-bit PNG using OpenCV (normalized 0..255)
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
        uint8_t v8 = static_cast<uint8_t>(std::lround(norm * 255.0));
        img8.at<uint8_t>(i / width, i % width) = v8;
    }
    cv::imwrite(base_path + ".png", img8);
}

int main(int argc, char** argv) {
    std::string path = (argc > 1 ? argv[1] : std::string("examples/dflat.txt"));
    std::vector<short> depth;
    if (!load_flat_data(path, depth)) return 1;

    // reshape constants
    const int n_frames = 8;
    const int height = 704;
    const int width = 1280;
    const int frame_pixels = height * width;
    if ((int)depth.size() != n_frames * frame_pixels) {
        std::cerr << "Data size mismatch: expected " << n_frames*frame_pixels
                  << " elements, got " << depth.size() << std::endl;
        return 1;
    }

    // Frame-level TRVL test: first frame only
    {
        short* first_frame = depth.data();
        int frame_size = frame_pixels;
        trvl::EncoderTRVL encoder(frame_size, 10, 2);
        trvl::DecoderTRVL decoder(frame_size);
        encoder.setKeyframe(true);

        // save original frame visualization
        save_depth_frame_visuals(first_frame, width, height, "trvl_frame_before");

        // measure frame encode time
        auto start_fenc = std::chrono::high_resolution_clock::now();
        auto compressed = encoder.encode(first_frame, true);
        auto end_fenc   = std::chrono::high_resolution_clock::now();
        std::cout << "Frame encode time: "
                  << std::chrono::duration<double, std::milli>(end_fenc - start_fenc).count()
                  << " ms\n";

        // measure frame decode time
        auto start_fdec   = std::chrono::high_resolution_clock::now();
        auto decompressed = decoder.decode(compressed.data(), true);
        auto end_fdec     = std::chrono::high_resolution_clock::now();
        std::cout << "Frame decode time: "
                  << std::chrono::duration<double, std::milli>(end_fdec - start_fdec).count()
                  << " ms\n";

        if ((int)decompressed.size() != frame_size) {
            std::cerr << "Decoded size mismatch: " << decompressed.size()
                      << " vs " << frame_size << std::endl;
            return 1;
        }
        for (int i = 0; i < frame_size; ++i) {
            if (decompressed[i] != first_frame[i]) {
                std::cerr << "Value mismatch at index " << i << ": " << decompressed[i]
                          << " vs " << first_frame[i] << std::endl;
                return 1;
            }
        }

        save_depth_frame_visuals(decompressed.data(), width, height, "trvl_frame_after");
        std::cout << "TRVL frame encoding/decoding test passed." << std::endl;
    }

    // Video-level TRVL test: all frames
    {
        int keyframe_interval = 1;
        // Initialize video-level TRVL encoder and decoder
        std::cout << "Testing TRVL video encoding/decoding..." << std::endl;
        std::cout << "Initializing TRVL video encoder..." << std::endl;
        trvl::VideoEncoderTRVL venc(keyframe_interval, frame_pixels, 10, 2);
        std::cout << "Initializing TRVL video decoder..." << std::endl;
        trvl::VideoDecoderTRVL vdec(keyframe_interval, frame_pixels);
        // Set the number of frames for encoding/decoding
        venc.setNumFrames(n_frames);
        vdec.setNumFrames(n_frames);

        // Encode full video buffer at once
        // measure video encode time
        auto start_venc = std::chrono::high_resolution_clock::now();
        auto compressed_video = venc.encode(depth.data());
        auto end_venc   = std::chrono::high_resolution_clock::now();
        std::cout << "Video encode time: "
                  << std::chrono::duration<double, std::milli>(end_venc - start_venc).count()
                  << " ms\n";

        // Prepare visuals for original frames
        for (int f = 0; f < n_frames; ++f) {
            short* frame_ptr = depth.data() + f * frame_pixels;
            save_depth_frame_visuals(frame_ptr, width, height, "trvl_video_before_f" + std::to_string(f));
        }

        // Decode video
        std::vector<char*> video_bytes;
        video_bytes.reserve(n_frames);
        for (auto& buf : compressed_video) video_bytes.push_back(buf.data());
        // measure video decode time
        auto start_vdec = std::chrono::high_resolution_clock::now();
        auto decoded    = vdec.decode(video_bytes);
        auto end_vdec   = std::chrono::high_resolution_clock::now();
        std::cout << "Video decode time: "
                  << std::chrono::duration<double, std::milli>(end_vdec - start_vdec).count()
                  << " ms\n";

        int total = n_frames * frame_pixels;
        if ((int)decoded.size() != total) {
            std::cerr << "Decoded video size mismatch: " << decoded.size()
                      << " vs " << total << std::endl;
            return 1;
        }

        // Validate and visualize each decoded frame
        for (int f = 0; f < n_frames; ++f) {
            const short* decoded_ptr = decoded[f].data() + f * frame_pixels;
            const short* original = depth.data() + f * frame_pixels;
            for (int i = 0; i < frame_pixels; ++i) {
                if (decoded_ptr[i] != original[i]) {
                    std::cerr << "Video mismatch at frame " << f << ", idx " << i
                              << ": " << decoded_ptr[i] << " vs " << original[i] << std::endl;
                    return 1;
                }
            }
            save_depth_frame_visuals(decoded_ptr, width, height, "trvl_video_after_f" + std::to_string(f));
        }
        std::cout << "TRVL video encoding/decoding test passed." << std::endl;
    }

    return 0;
}
