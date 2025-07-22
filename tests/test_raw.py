import numpy as np
import sys
from termcolor import colored
import argparse
import traceback

try:
    import fdcomp
except ImportError:
    print(colored("ERROR: ", 'red'), 'fdcomp module not found. Please install it first.')
    sys.exit(1)

def load(path: str = './examples/depth.npz'):
    depth = np.load(path)['depth']
    return depth

def test_raw_frame(path: str = './examples/depth.npz'):
    print(colored("\nTest: raw frame encoding and decoding.", "blue"))
    try:
        depth = load(path)[0]
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False

    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderRaw()
        print(colored("SUCCESS: ", 'green'), 'EncoderRaw initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderRaw:\n{e}')
        return False

    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderRaw()
        print(colored("SUCCESS: ", 'green'), 'DecoderRaw initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderRaw:\n{e}')
        return False

    ##-- Encoding --##
    try:
        compressed = encoder.encode(depth)
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        return False

    ##-- Decoding --##
    try:
        decoded = decoder.decode(compressed)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding data:\n{e}')
        return False

    ##-- Validation --##
    if not np.array_equal(depth, decoded):
        print(colored("ERROR: ", 'red'), "Decoded data does not match original.")
        return False

    print(colored("All raw frame tests passed!", 'green'))
    return True

def test_raw_video(path: str = './examples/depth.npz'):
    print(colored("\nTest: raw video encoding and decoding.", "blue"))
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False

    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderRawVideo()
        print(colored("SUCCESS: ", 'green'), 'EncoderRawVideo initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderRawVideo:\n{e}')
        return False

    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderRawVideo()
        print(colored("SUCCESS: ", 'green'), 'DecoderRawVideo initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderRawVideo:\n{e}')
        return False

    ##-- Encoding --##
    try:
        compressed = encoder.encode(depth)
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        return False

    ##-- Decoding --##
    try:
        decoded = decoder.decode(compressed)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding data:\n{e}')
        return False

    ##-- Validation --##
    if not np.array_equal(depth, decoded):
        print(colored("ERROR: ", 'red'), "Decoded video data does not match original.")
        return False

    print(colored("All raw video tests passed!", 'green'))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test raw Encoder/Decoder")
    parser.add_argument('--path', default='./examples/depth.npz', type=str)
    args = parser.parse_args()
    success = True
    success &= test_raw_frame(path=args.path)
    success &= test_raw_video(path=args.path)
    if success:
        print(colored("All tests passed successfully!", 'green'))
        sys.exit(0)
    else:
        print(colored("Some tests failed. Please check the output for details.", 'red'))
        sys.exit(1)
