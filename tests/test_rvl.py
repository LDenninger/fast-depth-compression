import numpy as np
import gnuplotlib as gp
import sys

from termcolor import colored
import argparse
import traceback

import time

try:
    import fdcomp
except ImportError:
    print(colored("ERROR: ", 'red'), 'fdcomp module not found. Please install it first.')
except Exception as e:
    print(colored("ERROR: ", 'red'), f'An error occurred while importing fdcomp:\n{e}')
    traceback.print_exc()


def load(path: str = './examples/depth.npz'):
    depth = np.load(path)['depth']
    return depth


def test_rvl_frame(path: str = './examples/depth.npz'):
    print(colored("\nTest: RVL frame encoding and decoding.", "blue"))
    try:
        depth = load(path)[0]
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False

    height, width = depth.shape
    frame_size = width * height
    byte_size = depth.nbytes

    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderRVL(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'EncoderRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderRVL:\n{e}')
        return False

    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderRVL(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'DecoderRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderRVL:\n{e}')
        traceback.print_exc()
        return False

    ##-- Encoding --##
    try:
        start_time = time.perf_counter()
        data_compressed = encoder.encode(depth)
        end_time = time.perf_counter()
        compression_time = (end_time - start_time)*1000
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        traceback.print_exc()
        return False

    ##-- Decoding --##
    try:
        start_time = time.perf_counter()
        data_decoded = decoder.decode(data_compressed)
        end_time = time.perf_counter()
        decoding_time = (end_time - start_time)*1000
        data_decoded = data_decoded.view(np.float16)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding data:\n{e}')
        traceback.print_exc()
        return False

    ##-- Validation --##
    depth_flat = depth.flatten()
    if depth_flat.shape[0] != data_decoded.shape[0]:
        print(colored("ERROR: ", 'red'), "Decoded data shape mismatch.")
        return False
    nbytes_compressed = len(data_compressed)
    byte_size_mb = byte_size / (1024 * 1024)
    nbytes_compressed_mb = nbytes_compressed / (1024 * 1024)
    compression_ratio = byte_size / nbytes_compressed
    l2_norm = np.linalg.norm(depth_flat - data_decoded)
    total_loss = np.sum(np.abs(depth_flat - data_decoded))

    print(colored("Results:", 'green'))
    print(f" Depth shape:....... {depth.shape}")
    print(f" Encoding time:..... {compression_time:.2f} ms")
    print(f" Decoding time:..... {decoding_time:.2f} ms")
    print(f" Original size:..... {byte_size_mb:.2f} MB")
    print(f" Compressed size:... {nbytes_compressed_mb:.2f} MB")
    print(f" Compression rate:.. {compression_ratio:.2f}")
    print(f" Compression loss:.. L2={l2_norm:.6f}, Total={total_loss:.4f}")

    return True


def test_rvl_video(path: str = './examples/depth.npz'):
    print(colored("\nTest: RVL video encoding and decoding.", "blue"))
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False

    height, width = depth.shape[1], depth.shape[2]
    frame_size = width * height
    shape = depth.shape
    byte_size = depth.nbytes

    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderRVLVideo(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'EncoderRVLVideo initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderRVLVideo:\n{e}')
        traceback.print_exc()
        return False

    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderRVLVideo(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'DecoderRVLVideo initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderRVLVideo:\n{e}')
        traceback.print_exc()
        return False

    ##-- Encoding --##
    try:
        start_time = time.perf_counter()
        compressed_data = encoder.encode(depth, verbose=True)
        end_time = time.perf_counter()
        compression_time = (end_time - start_time)*1000
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        traceback.print_exc()
        return False

    ##-- Decoding --##
    try:
        start_time = time.perf_counter()
        data_decoded = decoder.decode(compressed_data, verbose=True)
        end_time = time.perf_counter()
        decoding_time = (end_time - start_time)*1000
        data_decoded = data_decoded.view(np.float16)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding data:\n{e}')
        traceback.print_exc()
        return False

    ##-- Validation --##
    success = True
    nbytes_compressed = sum(len(d) for d in compressed_data)
    byte_size_mb = byte_size / (1024 * 1024)
    nbytes_compressed_mb = nbytes_compressed / (1024 * 1024)
    compression_ratio = byte_size / nbytes_compressed

    if data_decoded.shape[-1] != frame_size:
        print(colored("ERROR: ", 'red'), "Decoded frame size mismatch.")
        return False
    data_decoded = data_decoded.reshape(shape)

    decoded_is_nan = np.isnan(data_decoded)
    if np.any(decoded_is_nan):
        print(colored("ERROR: ", 'red'), f'Decoded data contains NaNs ({np.sum(decoded_is_nan)}).')
        success = False

    frame_wise_l2, frame_wise_total = [], []
    for i in range(shape[0]):
        valid = ~decoded_is_nan[i]
        l2 = np.linalg.norm(depth[i][valid] - data_decoded[i][valid])
        tot = np.sum(np.abs(depth[i][valid] - data_decoded[i][valid]))
        frame_wise_l2.append(l2)
        frame_wise_total.append(tot)

    frame_wise_l2 = np.array(frame_wise_l2)

    l2_norm = np.linalg.norm(depth[~decoded_is_nan] - data_decoded[~decoded_is_nan])
    total_loss = np.sum(np.abs(depth[~decoded_is_nan] - data_decoded[~decoded_is_nan]))

    print(colored("Results:", 'green' if success else 'red'))
    print(f" Depth shape:....... {depth.shape}")
    print(f" Encoding time:..... {compression_time:.2f} ms")
    print(f" Decoding time:..... {decoding_time:.2f} ms")
    print(f" Original size:..... {byte_size_mb:.2f} MB")
    print(f" Compressed size:... {nbytes_compressed_mb:.2f} MB")
    print(f" Compression rate:.. {compression_ratio:.2f}")
    print(f" Compression loss:.. L2={l2_norm:.6f}, Total={total_loss:.4f}")

    x = np.arange(shape[0])
    gp.plot(x, frame_wise_l2,
            title='Frame-wise L2 Norm', xlabel='Frame Index', ylabel='L2 Norm',
            _with    = 'lines',
            terminal = 'dumb 60,20',
            unset    = 'grid')

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RVL Encoder/Decoder")
    parser.add_argument('--path', default='./examples/depth.npz', type=str)
    args = parser.parse_args()
    print("Testing RVL Encoder/Decoder...")
    success = True
    success &= test_rvl_frame(path=args.path)
    success &= test_rvl_video(path=args.path)
    if success:
        print(colored("All tests passed successfully!", 'green'))
        sys.exit(0)
    else:
        print(colored("Some tests failed. Please check the output for details.", 'red'))
        sys.exit(1)

