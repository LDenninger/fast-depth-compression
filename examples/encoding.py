import fdcomp
import numpy as np
import matplotlib.pyplot as plt

#import ipdb; ipdb.set_trace()
width, height = 640, 480
timesteps = 10

depth_arr = np.load("./depth.npz")['depth'][0]
width, height = depth_arr.shape[1], depth_arr.shape[0]
#video_example = np.random.randint(0, 1000, (height, width), dtype=np.int16)

encoder = fdcomp.EncoderTRVL(width*height, 10, 2, False)
decoder = fdcomp.DecoderTRVL(width*height, False)
result = encoder.encode(depth_arr, False)
#print(len(result))
#print(type(result))


result_dec = decoder.decode(result)
result_dec = np.reshape(result_dec, (height, width))
result_dec = result_dec.view(np.float16)
#import ipdb; ipdb.set_trace()

l2 = np.linalg.norm(depth_arr - result_dec)
print("TRVL L2 norm:", l2)

d0_norm = ((depth_arr.astype(np.float32) / depth_arr.max())*255).astype(np.uint8)
plt.imsave("ground_truth.png", d0_norm,  cmap='gray')

d0_dec = ((result_dec.astype(np.float32) / result_dec.max())*255).astype(np.uint8)
plt.imsave("result_trvl.png", d0_dec,  cmap='gray')



#print(result)






