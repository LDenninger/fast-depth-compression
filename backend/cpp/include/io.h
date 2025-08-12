#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include "base.h"  // for FrameEncoder, FrameDecoder

namespace fdcomp {

struct FileMeta {
    std::string path;
    std::vector<int> shape;    // [num_frames, height, width]
    std::string dtype;         // e.g. "short"
    std::string encoder;
    std::string data_add;      // e.g. keyframe list
};

// Write raw depth frames to disk using a FrameEncoder.
inline void dump(const std::string& filename,
                 const std::vector<short>& data,
                 int num_frames,
                 int height,
                 int width,
                 VideoEncoder* encoder,
                 const std::string& encoder_name,
                 const std::vector<int>& keyframes = {})
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for writing: " + filename);
    // Write header
    std::ostringstream hdr;
    hdr << "!" << "(" << num_frames << "," << height << "," << width << ")"
        << "; " << "short"
        << "; " << encoder_name
        << "; [";
    for (size_t i = 0; i < keyframes.size(); ++i) {
        if (i) hdr << ",";
        hdr << keyframes[i];
    }
    hdr << "]\n";
    ofs << hdr.str();
    // Configure and encode video in one go
    encoder->setFrameSize(height * width);
    encoder->setNumFrames(num_frames);
    std::vector<char> comp = encoder->encode(data.data(), num_frames);
    // Write entire encoded data
    ofs.write(comp.data(), comp.size());
}

// Decode a sequence of compressed blocks into raw depth frames
inline std::vector<short> loads(std::vector<std::vector<char>>& blocks,
                                FrameDecoder* decoder)
{
    int frame_size = decoder->getFrameSize();
    std::vector<short> output;
    output.reserve(static_cast<size_t>(frame_size) * blocks.size());
    for (auto& blk : blocks) {
        char* ptr = blk.data();
        std::vector<short> frame = decoder->decode(ptr);
        output.insert(output.end(), frame.begin(), frame.end());
    }
    return output;
}

// Save multi-frame depth data with a FrameEncoder or VideoEncoder
inline void save(std::string& filename,
                 std::vector<short>& data,
                 int num_frames,
                 int height,
                 int width,
                 VideoEncoder* encoder,
                 std::string& encoder_name,
                 std::vector<int>& keyframes = {})
{
    // reuse dump
    dump(filename, data, num_frames, height, width, encoder, encoder_name, keyframes);
}

// Load depth frames from disk, returns flat data + metadata
inline std::pair<std::vector<short>, FileMeta> load(const std::string& filename, VideoDecoder* decoder)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file for reading: " + filename);
    // parse header
    std::string header;
    std::getline(ifs, header);
    if (header.empty() || header[0] != '!') {
        throw std::runtime_error("Invalid header in file: " + filename);
    }
    // remove leading '!'
    header = header.substr(1);
    std::istringstream ss(header);
    std::string part;
    FileMeta meta;
    meta.path = filename;
    // shape
    std::getline(ss, part, ';');
    part.erase(0, part.find_first_not_of(" \t"));
    part.erase(part.find_last_not_of(" \t") + 1);
    part = part.substr(1, part.size()-2);
    std::istringstream sh(part);
    int n,h,w;
    char comma;
    sh >> n >> comma >> h >> comma >> w;
    meta.shape = {n,h,w};
    // dtype
    std::getline(ss, part, ';');
    meta.dtype = part;
    // encoder
    std::getline(ss, part, ';');
    meta.encoder = part;
    // data_add
    std::getline(ss, part);
    meta.data_add = part;
    // read blocks
    std::vector<std::vector<char>> blocks;
    while (true) {
        char len_buf[4];
        ifs.read(len_buf,4);
        if (!ifs) break;
        uint32_t len = (static_cast<uint8_t>(len_buf[0]) << 24)
                     | (static_cast<uint8_t>(len_buf[1]) << 16)
                     | (static_cast<uint8_t>(len_buf[2]) <<  8)
                     | (static_cast<uint8_t>(len_buf[3])      );
        std::vector<char> blk(len);
        if (len) ifs.read(blk.data(), len);
        blocks.push_back(std::move(blk));
    }
    // configure and decode full video
    decoder->setFrameSize(h * w);
    decoder->setNumFrames(n);
    std::vector<char*> ptrs;
    ptrs.reserve(blocks.size());
    for (auto &blk : blocks) {
        ptrs.push_back(blk.data());
    }
    std::vector<short> data = decoder->decode(ptrs);
    return {data, meta};
}

} // namespace fdcomp


