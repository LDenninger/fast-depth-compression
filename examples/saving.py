import fdcomp
import numpy as np
import imageio

# 1. Load & round-trip your depth map
depth_arr = np.load("./depth.npz")['depth']  # shape: (frames, H, W)
loaded = fdcomp.save(depth_arr, "depth_trvl.dep") or fdcomp.load("depth_trvl.dep")

print(f"Loaded depth: shape={loaded.shape}, dtype={loaded.dtype}")

# 2. Compute L2 per frame just to verify
diff = depth_arr.reshape(loaded.shape[0], -1) - loaded.reshape(loaded.shape[0], -1)
l2 = np.linalg.norm(diff, ord=2, axis=1)
print("L2 norm per frame:", l2)

# 3. Normalization helper: [0, max] → [0,255] uint8
def normalize_uint8(arr):
    f = arr.astype(np.float32)
    m = f.max()
    return ((f / m) * 255).clip(0, 255).astype(np.uint8)

# Normalize once
gt_norm     = normalize_uint8(depth_arr)
loaded_norm = normalize_uint8(loaded)

# 4. Write MP4s with imageio (uses ffmpeg under the hood)
def write_mp4(frames_uint8, path, fps=7):
    # imageio expects H×W×3 (RGB) or H×W (grayscale)
    # to be safe, stack grayscale into RGB
    rgb_seq = [np.stack([f]*3, axis=-1) for f in frames_uint8]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in rgb_seq:
            writer.append_data(frame)

write_mp4(gt_norm,     "depth_gt.mp4",     fps=7)
write_mp4(loaded_norm, "depth_trvl.mp4", fps=7)

# 5. Compute difference, normalize, and write a “diff” video
diff_arr = np.abs(gt_norm.astype(np.int16) - loaded_norm.astype(np.int16)).astype(np.uint8)

def write_diff_mp4(diff_uint8, path, fps=7):
    # Stack grayscale diff into RGB
    rgb_diff_seq = [np.stack([f]*3, axis=-1) for f in diff_uint8]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in rgb_diff_seq:
            writer.append_data(frame)

write_diff_mp4(diff_arr, "depth_diff.mp4", fps=7)


print("🎥 Videos written to depth_gt.mp4 and depth_trvl.mp4")
