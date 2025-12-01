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

binary_mask = create_binary_mask('testing_binary_images/IMG_4.jpg', 'testing_binary_images/IMG_0.jpg', threshold=0.4)
io.imsave('binary_masks/binary_mask4.jpeg', img_as_ubyte(binary_mask))




