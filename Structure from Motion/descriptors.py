from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import glob
from imageio import imread, imsave
from mpl_toolkits.axes_grid1 import ImageGrid

from skimage.color import rgb2gray
import skimage.io as io
from skimage.feature import plot_matches
from skimage.feature import ORB, SIFT

from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
from skimage.feature import match_descriptors

def plot_imageset(images, figsize=(12, 10)):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1)

    for ax, im in zip(grid, images):
        ax.set_axis_off()
        if len(im.shape) < 3:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im)

    plt.show()
    

# pano_paths = sorted(glob.glob('images/img0*'))

# pano_imgs = [imread(path) for path in pano_paths]

# plot_imageset(pano_imgs)

# # Convert to grayscale
# image0, image1, pano2 = [rgb2gray(im) for im in pano_imgs]
# plot_imageset([image0, image1, pano2], figsize=(12, 10))

def keypoints(image0,image1):
    sift = SIFT()
    sift.detect_and_extract(image0)
    keypoints0 = sift.keypoints
    descriptors0 = sift.descriptors
    
    sift.detect_and_extract(image1)
    keypoints1 = sift.keypoints
    descriptors1 = sift.descriptors
    
    matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20, 20))

    plt.gray()
    
    plot_matches(ax, image0, image1, keypoints0, keypoints1, matches01)
    ax.axis('off')
    ax.set_title("Output of matcher: 0->1")
    
    src = keypoints0[matches01[:, 0]][:, ::-1]
    dst = keypoints1[matches01[:, 1]][:, ::-1]
    
    model_robust01, inliers01 = ransac((src, dst), ProjectiveTransform,
                        min_samples=4, residual_threshold=2, max_trials=300)
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20, 20))

    plt.gray()
    
    plot_matches(ax, image0, image1, keypoints0, keypoints1, matches01[inliers01])
    ax.axis('off')
    ax.set_title("Output of inliers: 0->1")
    
    return keypoints0, keypoints1, matches01, inliers01
    
    
    
    
 

