import numpy as np

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

    depth = np.load(path)['depth'][0]

    return depth

def test_trvl(path: str = './examples/depth.npz'):
    print("Testing TRVL Encoder/Decoder...")
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return
    
    height, width = depth.shape[0], depth.shape[1] 
    frame_size = width * height

    byte_size = depth.nbytes
    
    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderTRVL(frame_size, 10, 2, False)
        print(colored("SUCCESS: ", 'green'), 'EncoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderTRVL:\n{e}')
        return
    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderTRVL(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'DecoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderTRVL:\n{e}')
        return


    ##-- Encoding --##
    try:
        start_time = time.perf_counter()
        data_compressed = encoder.encode(depth)
        end_time = time.perf_counter()
        compression_time = (end_time - start_time)*1000  
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        return
    
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
        return
    
    ##-- Validation --##
    nbytes_compressed = len(data_compressed)
    byte_size_mb = byte_size / (1024 * 1024)
    nbytes_compressed_mb = nbytes_compressed / (1024 * 1024)
    compression_ratio =  byte_size / nbytes_compressed

    depth_flat = depth.flatten()
    if depth_flat.shape[0] != data_decoded.shape[0]:
        print(colored("ERROR: ", 'red'), "Decoded data shape does not match original depth data shape.")
        return

    l2_norm = np.linalg.norm(depth.flatten() - data_decoded)
    total_loss = np.sum(np.abs(depth.flatten() - data_decoded))

    print(colored("Results:", 'green'))
    print(f" Depth shape:....... {depth.shape}")
    print(f" Encoding time:..... {compression_time:.2f} ms")
    print(f" Decoding time:..... {decoding_time:.2f} ms")
    print(f" Original size:..... {byte_size_mb:.2f} MB")
    print(f" Compressed size:... {nbytes_compressed_mb:.2f} MB")
    print(f" Compression rate:.. {compression_ratio:.2f}")
    print(f" Compression loss:.. L2={l2_norm:.6f}, Total={total_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TRVL Encoder/Decoder")
    parser.add_argument('--path', default='./examples/depth.npz', type=str, help='Path to the depth data file')
    args = parser.parse_args()
    test_trvl(path=args.path)

