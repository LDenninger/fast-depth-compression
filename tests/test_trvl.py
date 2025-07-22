import numpy as np

import gnuplotlib as gp

from termcolor import colored
import argparse
import traceback
import sys
import cv2


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

def test_trvl_frame(path: str = './examples/depth.npz'):
    try:
        depth = load(path)[0]
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False
    
    height, width = depth.shape[0], depth.shape[1] 
    frame_size = width * height

    byte_size = depth.nbytes
    
    ##-- Encoder Initialization --##
    try:
        encoder = fdcomp.EncoderTRVL(frame_size, 10, 2, False)
        print(colored("SUCCESS: ", 'green'), 'EncoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderTRVL:\n{e}')
        return False
    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderTRVL(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'DecoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderTRVL:\n{e}')
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
        return False
    
    ##-- Validation --##
    nbytes_compressed = len(data_compressed)
    byte_size_mb = byte_size / (1024 * 1024)
    nbytes_compressed_mb = nbytes_compressed / (1024 * 1024)
    compression_ratio =  byte_size / nbytes_compressed

    depth_flat = depth.flatten()
    if depth_flat.shape[0] != data_decoded.shape[0]:
        print(colored("ERROR: ", 'red'), "Decoded data shape does not match original depth data shape.")
        return False

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

    return True

def test_trvl_video(path: str = './examples/depth.npz', keyframe_stride: int = 10, change_threshold: int = 10,
                     save: bool = False):
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
        encoder = fdcomp.EncoderTRVLVideo(frame_size, change_threshold, 2, keyframe_stride, True)
        print(colored("SUCCESS: ", 'green'), 'EncoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing EncoderTRVL:\n{e}')
        return False
    ##-- Decoder Initialization --##
    try:
        decoder = fdcomp.DecoderTRVLVideo(frame_size, False)
        print(colored("SUCCESS: ", 'green'), 'DecoderTRVL initialized successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while initializing DecoderTRVL:\n{e}')
        return False


    ##-- Encoding --##
    try:
        start_time = time.perf_counter()
        data_compressed, keyframes = encoder.encode(depth)
        end_time = time.perf_counter()
        compression_time = (end_time - start_time)*1000  
        print(colored("SUCCESS: ", 'green'), 'Data encoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while encoding data:\n{e}')
        return False
    
    ##-- Decoding --##
    try:
        start_time = time.perf_counter()
        data_decoded = decoder.decode(data_compressed, keyframes)
        end_time = time.perf_counter()
        decoding_time = (end_time - start_time)*1000
        data_decoded = data_decoded.view(np.float16)

        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding data:\n{e}')
        return False
    
    ##-- Validation --##
    success = True

    nbytes_compressed = sum([len(d) for d in data_compressed])
    byte_size_mb = byte_size / (1024 * 1024)
    nbytes_compressed_mb = nbytes_compressed / (1024 * 1024)
    compression_ratio =  byte_size / nbytes_compressed

    if data_decoded.shape[-1] != frame_size:
        print(colored("ERROR: ", 'red'), "Decoded frame size does not match the original frame size.")
        print(f" Original frame size: {frame_size}, Decoded frame size: {data_decoded.shape[-1]}")
        return False
    
    data_decoded = data_decoded.reshape(shape)

    decoded_is_nan = np.isnan(data_decoded)
    if np.any(decoded_is_nan):
        print(colored("ERROR: ", 'red'), "Decoded data contains NaN values.")
        print(f" NaN indices: {np.where(decoded_is_nan)}")
        print(f" NaN count: {np.sum(decoded_is_nan)}")
        success = False
    
    #import ipdb; ipdb.set_trace()

    frame_wise_mse = []
    for i in range(data_decoded.shape[0]):
        frame = data_decoded[i]
        if decoded_is_nan[i].any():
            print(f" -> Nan encountered in frame {i}")
        
        valid_pixels = ~decoded_is_nan[i]

        mse = np.mean((depth[i][valid_pixels].flatten() - frame[valid_pixels].flatten())**2)
        frame_wise_mse.append(mse)

    frame_wise_mse = np.array(frame_wise_mse)
    mse_all = np.mean((depth[~decoded_is_nan].flatten() - data_decoded[~decoded_is_nan].flatten())**2)
    total_loss = np.sum(np.abs(depth[~decoded_is_nan] - data_decoded[~decoded_is_nan]))

    print(colored("Results:", 'green' if success else 'red'))
    print(f" Depth shape:....... {depth.shape}")
    print(f" Encoding time:..... {compression_time:.2f} ms")
    print(f" Decoding time:..... {decoding_time:.2f} ms")
    print(f" Original size:..... {byte_size_mb:.2f} MB")
    print(f" Compressed size:... {nbytes_compressed_mb:.2f} MB")
    print(f" Compression rate:.. {compression_ratio:.2f}")
    print(f" Compression loss:.. MSE={mse_all:.6f}, Total={total_loss:.4f}")
    #print(f" L2 Norm (frame):... {', '.join([f'{l2:.6f}' for l2 in frame_wise_l2])}")
    #print(f" Total Loss (frame): {', '.join([f'{total:.4f}' for total in frame_wise_total])}")

    if mse_all > 0.0:
        x = np.arange(data_decoded.shape[0])
        gp.plot(x, frame_wise_mse, title='Frame-wise MSE',
                _with    = 'lines',
                terminal = 'dumb 80,30',
                unset    = 'grid')
        
    if save:
        diff_image = np.log(np.abs(depth - data_decoded)+1)
        diff_image =  (diff_image / np.max(diff_image))*255
        diff_image = diff_image.astype(np.uint8)[..., np.newaxis]

        depth_norm = (depth / np.max(depth))*255
        data_decoded_norm = (data_decoded / np.max(data_decoded))*255

        save_arr = np.concatenate([depth_norm, data_decoded_norm], axis=2).astype(np.uint8)[..., np.newaxis]
        # Save as video
        save_path = f"./tests/trvl_kf{keyframe_stride}_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 7.0, (width*2, height))
        for frame in save_arr:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        out.release() 

        save_path = f"./tests/trvl_kf{keyframe_stride}_diff.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 7.0, (width, height))
        for frame in diff_image:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TRVL Encoder/Decoder")
    parser.add_argument('--path', default='./examples/depth.npz', type=str, help='Path to the depth data file')
    parser.add_argument('--save', action="store_true", default=False, help="Save intermediate results for debugging")
    args = parser.parse_args()
    print("Testing TRVL Encoder/Decoder...")

    success = True
    print(colored("\nTest: TRVL frame encoding and decoding.", "blue"))
    success &= test_trvl_frame(path=args.path)

    keyframe_stride = 0
    change_threshold = 1
    print(colored(f"\nTest: TRVL video encoding and decoding. keyframe stride: {keyframe_stride}, change threshold: {change_threshold}", "blue"))
    success &= test_trvl_video(path=args.path, keyframe_stride=keyframe_stride, save=args.save)
    
    keyframe_stride = 10
    change_threshold = 1
    print(colored(f"\nTest: TRVL video encoding and decoding. keyframe stride: {keyframe_stride}, change threshold: {change_threshold}", "blue"))
    success &= test_trvl_video(path=args.path, keyframe_stride=keyframe_stride, save=args.save)
    
    keyframe_stride = 10
    change_threshold = 10
    print(colored(f"\nTest: TRVL video encoding and decoding. keyframe stride: {keyframe_stride}, change threshold: {change_threshold}", "blue"))
    success &= test_trvl_video(path=args.path, keyframe_stride=keyframe_stride, save=args.save)
    
    keyframe_stride = 20
    change_threshold = 10
    print(colored(f"\nTest: TRVL video encoding and decoding. keyframe stride: {keyframe_stride}, change threshold: {change_threshold}", "blue"))
    success &= test_trvl_video(path=args.path, keyframe_stride=keyframe_stride, save=args.save)


    keyframe_stride = 50
    change_threshold = 10
    print(colored(f"\nTest: TRVL video encoding and decoding. keyframe stride: {keyframe_stride}, change threshold: {change_threshold}", "blue"))
    success &= test_trvl_video(path=args.path, keyframe_stride=keyframe_stride, save=args.save)

    if success:
        print(colored("All tests passed successfully!", 'green'))
        sys.exit(0)
    else:
        print(colored("Some tests failed. Please check the output for details.", 'red'))
        sys.exit(1)

