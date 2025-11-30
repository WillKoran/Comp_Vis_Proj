import numpy as np
from skimage import io, img_as_ubyte, morphology
from scipy import ndimage # Required for filling holes

def create_binary_mask(image_path, background_path, threshold):
    # read images
    image = io.imread(image_path, as_gray=True)
    background = io.imread(background_path, as_gray=True)

    # create the intial binary mask
    binary_image = img_as_ubyte(image)

    # set up the binary mask
    difference = np.abs(image - background)
    binary_image = difference > threshold
    
    initial_mask = morphology.binary_closing(binary_image, morphology.disk(1))
    binary_mask = ndimage.binary_fill_holes(initial_mask)

    return binary_mask

binary_mask = create_binary_mask('IMG_0001.jpeg', 'IMG_0002.jpeg', threshold=0.1)
io.imsave('binary_mask.jpeg', img_as_ubyte(binary_mask))




