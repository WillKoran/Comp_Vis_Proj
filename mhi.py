import numpy as np
import skimage
from skimage import io, img_as_ubyte


def create_mhi(image_collection, threshold, history_duration):
    first_frame = image_collection[0]
    mhi = np.zeros(first_frame.shape, dtype=np.float32)
    
    num_frames = len(image_collection)
    for i in range(1, num_frames):
        tau = i
        
        curr_frame = image_collection[i].astype(np.float32)
        prev_frame = image_collection[i - 1].astype(np.float32)
        
        frame_diff = np.abs(curr_frame - prev_frame)
        motion_mask = frame_diff > threshold
        mhi[motion_mask] = tau

        no_motion_mask = ~motion_mask
        expired_limit = tau - history_duration
        
        decay_mask = np.logical_and(no_motion_mask, mhi < expired_limit)
        
        mhi[decay_mask] = 0

    return mhi

motion_collection = skimage.io.imread_collection('motion_images/aerobic-*.bmp')
T2 = 5
N = 22

mhi = create_mhi(motion_collection, T2, N)
mhi_normalized = mhi / np.max(mhi)
io.imsave('mhi_output.jpg', img_as_ubyte(mhi_normalized))


