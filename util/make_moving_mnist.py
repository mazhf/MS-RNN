import math
import os
import sys
import cv2
import numpy as np
from PIL import Image


###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

# helper functions
def arr_from_img(im, mean=0, std=1):
    '''
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    '''

    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


# loads mnist from web on demand
def load_dataset(training=True):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images('train-images-idx3-ubyte.gz')
    return load_mnist_images('t10k-images-idx3-ubyte.gz')


def generate_moving_mnist(training, shape=(64, 64), num_frames=30, num_images=100, original_size=28, nums_per_image=2):
    '''

    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_images: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.

    Returns:
        Dataset of np.uint8 type with dimensions num_frames x num_images x new_width x new_height

    '''
    mnist = load_dataset(training)
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_images, num_frames, 1, width, height), dtype=np.uint8)

    for img_idx in range(num_images):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=nums_per_image) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing two PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(get_image_from_array(mnist, r, mean=0)).resize((original_size, original_size),
                                                                                       Image.ANTIALIAS) \
                        for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            img = (canvas * 255).clip(0, 255).astype(np.uint8)
            img = img.squeeze(axis=0)
            img = cv2.flip(img, 0)  # 沿x轴翻转
            cols, rows = img.shape  # 逆时针270°
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 * 3, 1)
            dst = cv2.warpAffine(img, M, (rows, cols))
            dataset[img_idx, frame_idx, ...] = dst

    return dataset.squeeze(axis=2).transpose(1, 0, 2, 3)


def main(training, path, filetype='npy', frame_size=64, num_frames=30, num_images=100, original_size=28,
         nums_per_image=2):
    dat = generate_moving_mnist(training, shape=(frame_size, frame_size), num_frames=num_frames, num_images=num_images,
                                original_size=original_size, nums_per_image=nums_per_image)
    n = num_images * num_frames
    if filetype == 'npy':
        np.save(path, dat)
    elif filetype == 'jpg':
        for i in range(dat.shape[0]):
            Image.fromarray(get_image_from_array(dat, i, mean=0)).save(os.path.join(path, '{}.jpg'.format(i)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dest', type=str, dest='path', default='/home/mazhf/Spatiotemporal/moving-mnist-30')  # 不需要加.npy后缀
    parser.add_argument('--filetype', type=str, dest='filetype', default="npy")
    parser.add_argument('--training', type=bool, dest='training', default=True)
    parser.add_argument('--frame_size', type=int, dest='frame_size', default=64)
    parser.add_argument('--num_frames', type=int, dest='num_frames', default=30)  # length of each sequence
    parser.add_argument('--num_images', type=int, dest='num_images', default=15000)  # number of sequences to generate
    parser.add_argument('--original_size', type=int, dest='original_size', default=28)  # size of mnist digit within frame
    parser.add_argument('--nums_per_image', type=int, dest='nums_per_image', default=2)  # number of digits in each frame
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
