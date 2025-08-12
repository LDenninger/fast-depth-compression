#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>



class FrameEncoder {
public:
    // Construct with frame size
    explicit FrameEncoder(int frame_size)
        : frame_size_(frame_size) {}

    // Pure virtual: encode one depth frame into compressed bytes
    virtual std::vector<char> encode(short* depth_buffer) = 0;

    // Accessors
    int getFrameSize() { return frame_size_; }
    void setFrameSize(int frame_size) { frame_size_ = frame_size; }

protected:
    int frame_size_;
};
    


class VideoEncoder {
public:
    // Correct constructor name
    VideoEncoder(FrameEncoder* encoder){
        frame_size_ = encoder->getFrameSize();
        frame_encoder_ = encoder;
    }

    std::vector<std::vector<char>> encode(short* depth_buffer, int num_frames){
        setNumFrames(num_frames);
        return encode(depth_buffer);
    }

    std::vector<std::vector<char>> encode(short* depth_buffer) {

        if (frame_encoder_ == nullptr) {
            throw std::runtime_error("FrameEncoder needs to be initialized if using the default encode() method.");
        }
        // preallocate maximum possible size
        //size_t max_bytes = static_cast<size_t>(frame_size_) * num_frames_;
        std::vector<std::vector<char>> encoded_data;
        size_t offset = 0;
        for (int i = 0; i < num_frames_; i++) {
            short* frame_ptr = depth_buffer + i * frame_size_;
            std::vector<char> frame = frame_encoder_->encode(frame_ptr);
            encoded_data.push_back(frame);

        }
        // shrink to actual size
        return encoded_data;
    }

    void setFrameEncoder(FrameEncoder* encoder) {
        frame_encoder_ = encoder;
    }
    FrameEncoder* getFrameEncoder() {
        return frame_encoder_;
    }
    void setFrameSize(int frame_size) {
        frame_size_ = frame_size;
        if(frame_encoder_ != nullptr) {
            frame_encoder_->setFrameSize(frame_size);
        }
    }
    int getFrameSize() { return frame_size_; }
    void setNumFrames(int num_frames) {
        num_frames_ = num_frames;
    }
    int getNumFrames() { return num_frames_; }

protected:
    int frame_size_;
    int num_frames_;

    FrameEncoder* frame_encoder_;  // Pointer to FrameEncoder
};


class FrameDecoder {
public:
    // Construct with frame size (number of depth samples per frame)
    explicit FrameDecoder(int frame_size)
        : frame_size_(frame_size) {}

    virtual std::vector<short> decode(char* compressed_bytes) = 0;

    int getFrameSize() { return frame_size_; }
    void setFrameSize(int frame_size) { frame_size_ = frame_size; }

protected:
    int frame_size_;
};

class VideoDecoder {
public:
    // Takes ownership or reference of a FrameDecoder
    explicit VideoDecoder(FrameDecoder* decoder)
        : frame_decoder_(decoder)
        , frame_size_(decoder ? decoder->getFrameSize() : 0)
        , num_frames_(0)
    {}

    std::vector<std::vector<short>> decode(std::vector<std::vector<char>> video_bytes) {
        std::vector<char*> ptrs;
        ptrs.reserve(video_bytes.size());
        for (auto &buf : video_bytes) {
            ptrs.push_back(buf.data());
        }
        return decode(ptrs);
    }

    std::vector<std::vector<short>> decode(std::vector<char*> video_bytes) {
        if (!frame_decoder_) {
            throw std::runtime_error("FrameDecoder must be initialized before calling decode()");
        }
        size_t frame_chunk_size = video_bytes.size() / num_frames_;
        if (video_bytes.size() != num_frames_) {
            throw std::runtime_error("Must provide same number of encoed bytes as number of frames for default decode().");
        }

        std::vector<std::vector<short>> output;

        for (int i = 0; i < static_cast<int>(video_bytes.size()); ++i) {
            // slice out this frame’s compressed bytes
            char* frame_bytes = video_bytes.at(i);

            // decode to raw depth samples
            std::vector<short> decoded_frame = frame_decoder_->decode(frame_bytes);
            output.push_back(decoded_frame);
        }

        return output;
    }

    // Mutators
    void setFrameDecoder(FrameDecoder* decoder) {
        frame_decoder_ = decoder;
        if (decoder) {
            frame_size_ = decoder->getFrameSize();
        }
    }

    void setFrameSize(int frame_size) {
        frame_size_ = frame_size;
        if (frame_decoder_) {
            frame_decoder_->setFrameSize(frame_size);
        }
    }

    void setNumFrames(int num_frames) {
        num_frames_ = num_frames;
    }

    // Accessors
    int getFrameSize()  { return frame_size_; }
    int getNumFrames()  { return num_frames_; }
    FrameDecoder* getFrameDecoder() { return frame_decoder_; }

protected:
    FrameDecoder*  frame_decoder_;
    int            frame_size_;
    int            num_frames_;
};