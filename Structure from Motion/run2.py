from pathlib import Path
import json
import numpy as np
from numpy.linalg import inv, svd, det, norm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image
import glob
from imageio import imread, imsave
from skimage.color import rgb2gray
import open3d as o3d
from vis import vis_3d, o3d_pc, draw_camera
from mpl_interactive import Visualizer as TwoViewVis

from sfm import (
    t_and_R_from_pose_pair,
    essential_from_t_and_R,
    F_from_K_and_E,
    E_from_K_and_F,
    t_and_R_from_essential,
    disambiguate_four_chirality_by_triangulation,
    triangulate,
    normalized_eight_point_algorithm,
    eight_point_algorithm,
    bundle_adjustment,
    align_B_to_A
)

from render import (
    compute_intrinsics, compute_extrinsics, as_homogeneous, homogenize
)

from descriptors import (plot_imageset, keypoints)

DATA_ROOT = Path("./data")
path = 'images/img0*'


def read_view_image(i):
    fname = DATA_ROOT / "sfm" / f"view_{i}.png"
    img = np.array(Image.open(fname))
    return img

def read_image(path):
    paths = sorted(glob.glob((path)))
    
    imgs = [imread(path) for path in paths]
    image0, image1 = [rgb2gray(im) for im in imgs]
    plot_imageset([image0, image1], figsize=(12, 10))
    return image0, image1


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


class Problems():
    def __init__(self):
       
        image0, image1 = read_image(path)
        
        keypoints0, keypoints1, matches01, inliers01 = keypoints(image0, image1)
        
        img_w, img_h = 800, 600
        fov = 53.7
        K = compute_intrinsics(img_w / img_h, fov, img_h)

        self.K = K
        self.img_w, self.img_h = img_w, img_h
        self.image0, self.image1 = image0, image1
        self.keypoints0, self.keypoints1 = keypoints0, keypoints1
        self.matches01 = matches01[inliers01]



    def sfm_pipeline(self, use_BA=False, draw_config=False, final_vis=True):

        K = self.K
        img_w, img_h = self.img_w, self.img_h

        x1s = as_homogeneous(self.keypoints0[self.matches01[:, 0]])
        x2s = as_homogeneous(self.keypoints1[self.matches01[:, 1]])

        full_K = K
        K = K[:3, :3]
        # F = normalized_eight_point_algorithm(x1s, x2s, img_w, img_h)

        F = eight_point_algorithm(x1s, x2s)
        E = E_from_K_and_F(K, F)
        print(E)

        four_tR_hypothesis = t_and_R_from_essential(E)
        #for (t, R) in four_tR_hypothesis:
            #print(t)
            #print(R)

        p1, p2, t, R = disambiguate_four_chirality_by_triangulation(
            four_tR_hypothesis, x1s, x2s, full_K, draw_config=draw_config
        )
        pred_pts = triangulate(full_K @ inv(p1), x1s, full_K @ inv(p2), x2s)
        print(pred_pts.shape)
        
        if use_BA:
            p1, p2, pred_pts = bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts)


        if final_vis:
            red = (1, 0, 0)
            green = (0, 1, 0)
            blue = (0, 0, 1)

            vis_3d(
                1000, 1000,
                o3d_pc(pred_pts, green),
                draw_camera(K, p1, img_w, img_h, 10, blue),
                draw_camera(K, p2, img_w, img_h, 10, blue),
            )


def main():
    engine = Problems()
    engine.sfm_pipeline()

if __name__ == "__main__":
    main()
