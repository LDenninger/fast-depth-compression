import os
import numpy as np

import argparse
import traceback
from termcolor import colored

try:
    import fdcomp
except ImportError:
    print(colored("ERROR: ", 'red'), 'fdcomp module not found. Please install it first.')
except Exception as e:
    print(colored("ERROR: ", 'red'), f'An error occurred while importing fdcomp:\n{e}')
    traceback.print_exc()

TRVL_PATH = './examples/depth_trvl_tmp'
RVL_PATH  = './examples/depth_rvl_tmp'
RAW_PATH  = './examples/depth_raw_tmp'   # add raw path

def load(path: str = './examples/depth.npz'):
    depth = np.load(path)['depth']
    return depth

def test_saving(path: str = './examples/depth.npz'):
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False
    success = True

    print(colored("\nTest: Saving using TRVL encoder.", "blue"))
    try:
        fdcomp.save(depth, TRVL_PATH, 'trvl')
        print(colored("SUCCESS: ", 'green'), 'Data saved successfully using TRVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while saving data:\n{e}')
        traceback.print_exc()
        success = True
    
    print(colored("\nTest: Saving using RVL encoder.", "blue"))
    try:
        fdcomp.save(depth, RVL_PATH, 'rvl')
        print(colored("SUCCESS: ", 'green'), 'Data saved successfully using RVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while saving data:\n{e}')
        traceback.print_exc()
        success = True

    print(colored("\nTest: Saving using raw encoder.", "blue"))  # add raw save
    try:
        fdcomp.save(depth, RAW_PATH, 'raw')
        print(colored("SUCCESS: ", 'green'), 'Data saved successfully using raw encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while saving data (raw):\n{e}')
        traceback.print_exc()
        success = False

    return success

def test_loading(path: str = './examples/depth.npz'):
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False
    success = True
    depth_trvl, depth_rvl, depth_raw = None, None, None

    print(colored("\nTest: Loading using TRVL encoder.", "blue"))
    try:
        depth_trvl = fdcomp.load(TRVL_PATH, 'trvl')
        print(colored("SUCCESS: ", 'green'), 'Data loaded successfully using TRVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading data:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Loading using RVL encoder.", "blue"))
    try:
        depth_rvl = fdcomp.load(RVL_PATH, 'rvl')
        print(colored("SUCCESS: ", 'green'), 'Data loaded successfully using RVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading data:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Loading using raw encoder.", "blue"))  # add raw load
    try:
        depth_raw = fdcomp.load(RAW_PATH, 'raw')
        print(colored("SUCCESS: ", 'green'), 'Data loaded successfully using raw encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading data (raw):\n{e}')
        traceback.print_exc()
        success = False

    ##-- Validation --##
    print(colored("\nValidation:", "blue"))

    if depth_trvl is not None:
        if depth.shape != depth_trvl.shape:
            print(colored("ERROR: ", 'red'), f'TRVL loaded data shape {depth_trvl.shape} does not match original {depth.shape}')
            success = False
        mse_trvl = np.mean((depth - depth_trvl) ** 2)
        if mse_trvl > 1e-2:
            print(colored("ERROR: ", 'red'), f'TRVL loaded data has high MSE: {mse_trvl}')
            success = False

    if depth_rvl is not None:
        if depth.shape != depth_rvl.shape:
            print(colored("ERROR: ", 'red'), f'RVL loaded data shape {depth_rvl.shape} does not match original {depth.shape}')
            success = False

        mse_rvl = np.mean((depth - depth_rvl) ** 2)
        if mse_rvl > 1e-2:
            print(colored("ERROR: ", 'red'), f'RVL loaded data has high MSE: {mse_rvl}')
            success = False

    if depth_raw is not None:  # raw is lossless
        if depth.shape != depth_raw.shape:
            print(colored("ERROR: ", 'red'), f'Raw loaded data shape {depth_raw.shape} does not match original {depth.shape}')
            success = False
        if not np.array_equal(depth, depth_raw):
            print(colored("ERROR: ", 'red'), "Raw loaded data does not match original.")
            success = False

    if success:
        print(colored("SUCCESS: ", 'green'), 'All validations passed successfully.')
    return success

def test_dumping(path: str = './examples/depth.npz'):
    try:
        depth = load(path)
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while loading depth data:\n{e}')
        return False
    success = True
    frame_size = depth.shape[-1] * depth.shape[-2]

    print(colored("\nTest: Dumping using TRVL encoder.", "blue"))
    try:
        data_trvl, keyframes = fdcomp.dump(depth, 'trvl')
        print(colored("SUCCESS: ", 'green'), 'Data dumped successfully using TRVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while dumping data:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Dumping using RVL encoder.", "blue"))
    try:
        data_rvl = fdcomp.dump(depth, 'rvl')
        print(colored("SUCCESS: ", 'green'), 'Data dumped successfully using RVL encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while dumping data:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Dumping using raw encoder.", "blue"))  # add raw dump
    try:
        data_raw = fdcomp.dump(depth, 'raw')
        print(colored("SUCCESS: ", 'green'), 'Data dumped successfully using raw encoder.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while dumping data (raw):\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Loading from TRVL dump.", "blue"))
    try:
        data_trvl_dec = fdcomp.loads(data_trvl, 'trvl', keyframes=keyframes, frame_size=frame_size)
        data_trvl_dec = data_trvl_dec.view(np.float16).reshape(depth.shape)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully from TRVL dump.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding TRVL dump:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Loading from RVL dump.", "blue"))
    try:
        data_rvl_dec = fdcomp.loads(data_rvl, 'rvl', frame_size=frame_size)
        data_rvl_dec = data_rvl_dec.view(np.float16).reshape(depth.shape)
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully from RVL dump.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding RVL dump:\n{e}')
        traceback.print_exc()
        success = False

    print(colored("\nTest: Loading from raw dump.", "blue"))  # add raw loads
    try:
        data_raw_dec = fdcomp.loads(data_raw, 'raw')
        print(colored("SUCCESS: ", 'green'), 'Data decoded successfully from raw dump.')
    except Exception as e:
        print(colored("ERROR: ", 'red'), f'An error occurred while decoding raw dump:\n{e}')
        traceback.print_exc()
        success = False

    ##-- Validation --##
    print(colored("\nValidation:", "blue"))
    if data_trvl_dec is not None:
        if depth.shape != data_trvl_dec.shape:
            print(colored("ERROR: ", 'red'), f'TRVL decoded data shape {data_trvl_dec.shape} does not match original {depth.shape}')
            success = False
        mse_trvl = np.mean((depth - data_trvl_dec) ** 2)
        if mse_trvl > 1e-2:
            print(colored("ERROR: ", 'red'), f'TRVL decoded data has high MSE: {mse_trvl}')
            success = False

    if data_rvl_dec is not None:
        if depth.shape != data_rvl_dec.shape:
            print(colored("ERROR: ", 'red'), f'RVL decoded data shape {data_rvl_dec.shape} does not match original {depth.shape}')
            success = False

        mse_rvl = np.mean((depth - data_rvl_dec) ** 2)
        if mse_rvl > 1e-2:
            print(colored("ERROR: ", 'red'), f'RVL decoded data has high MSE: {mse_rvl}')
            success = False

    if 'data_raw_dec' in locals():
        if depth.shape != data_raw_dec.shape:
            print(colored("ERROR: ", 'red'), f'Raw decoded data shape {data_raw_dec.shape} does not match original {depth.shape}')
            success = False
        if not np.array_equal(depth, data_raw_dec):
            print(colored("ERROR: ", 'red'), "Raw decoded data does not match original.")
            success = False

    if success:
        print(colored("SUCCESS: ", 'green'), 'All validations passed successfully.')

    return success

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test fdcomp I/O operations")
    parser.add_argument('--path', default='./examples/depth.npz', type=str, help="Path to the depth map file")
    args = parser.parse_args()
    success = True

    success &= test_dumping(args.path)
    success &= test_saving(args.path)
    success &= test_loading(args.path)

    if success:
        print(colored("\nAll tests passed successfully!", "green"))
    else:
        print(colored("\nSome tests failed. Please check the output for details.", "red"))