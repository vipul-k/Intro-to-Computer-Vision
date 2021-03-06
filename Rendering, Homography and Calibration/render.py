import numpy as np
from numpy import cross, tan
from numpy.linalg import norm, inv


def normalize(v):
    return v / norm(v)


def camera_pose(eye, front, up):
    # pay attn to whether you are dealing with column vector / row vector
    # a numpy vector by convention is a row vector.

    z = normalize(-1 * front)
    x = normalize(cross(up, z))
    y = normalize(cross(z, x))

    # convert to col vector
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    eye = eye.reshape(-1, 1)

    pose = np.block([
        [x, y, z, eye],
        [0, 0, 0, 1]
    ])
    return pose


def compute_extrinsics(eye, front, up):
    pose = camera_pose(eye, front, up)
    world_2_cam = inv(pose)
    return world_2_cam


def compute_intrinsics(aspect_ratio, fov, img_height_in_pix):
    # aspect ratio is  w / h
    ndc = compute_proj_to_normalized(aspect_ratio, fov)

    # should ask students to do a clipping here with ndc;
    # anything beyond [-1, 1] should be discarded
    # this did not mention how to do z-clipping;

    ndc_to_img = compute_normalized_to_img_trans(aspect_ratio, img_height_in_pix)
    intrinsic = ndc_to_img @ ndc
    return intrinsic


def compute_proj_to_normalized(aspect, fov):
    # compared to standard OpenGL NDC intrinsic,
    # this skips the 3rd row treatment on z. hence the name partial_ndc
    # it's incomplete, but enough for now
    fov_in_rad = fov / 180 * np.pi
    t = tan(fov_in_rad / 2)  # this is the tan half fov
    partial_ndc_intrinsic = np.array([
        [1 / (t * aspect), 0, 0, 0],
        [0, 1 / t, 0, 0],
        [0, 0, -1, 0]  # copy the negative distance for division
    ])
    return partial_ndc_intrinsic


def compute_normalized_to_img_trans(aspect, img_height_in_pix):
    img_h = img_height_in_pix
    img_w = img_height_in_pix * aspect

    # note the OpenGL convention that (0, 0) sits at the center of the pixel;
    # hence the extra -0.5 translation
    # this is useful when you shoot rays through a pixel to the scene
    # and is part of the intrinsics standard on principal point
    ndc_to_img = np.array([
        [img_w / 2, 0, img_w / 2 - 0.5],
        [0, img_h / 2, img_h / 2 - 0.5],
        [0, 0, 1]
    ])

    img_y_coord_flip = np.array([
        [1, 0, 0],
        [0, -1, img_h - 1],  # note the -1
        [0, 0, 1]
    ])

    # the product of the above 2 matrices is equivalent to adding
    # - sign to the (1, 1) entry
    # you could have simply written
    # ndc_to_img = np.array([
    #     [img_w / 2, 0, img_w / 2 - 0.5],
    #     [0, -img_h / 2, img_h / 2 - 0.5],
    #     [0, 0, 1]
    # ])

    ndc_to_img = img_y_coord_flip @ ndc_to_img
    return ndc_to_img


def rays_through_pixels(K, pixel_coords):
    # pixel coords: [n, 2] pix locations
    K = K[0:3, 0:3]

    pixel_coords = as_homogeneous(pixel_coords)
    pixel_coords = pixel_coords.T  # [2+1, n], so that mat mult is on the left

    # this will give points with z = -1, which is exactly what you want since
    # your camera is facing the -ve z axis
    pts = inv(K) @ pixel_coords
    pts = pts.T
    # promote these as 3d points
    pts = as_homogeneous(pts)
    # rays start from origin
    rays = pts - np.array([0, 0, 0, 1])
    return rays


def homogenize(pts):
    # pts: [n, d], where last dim of the d is the diviser
    pts = pts / pts[:, -1].reshape(-1, 1)
    return pts


def as_homogeneous(pts):
    # pts: [n, d]
    n, d = pts.shape
    points = np.ones((n, d + 1))
    points[:, :d] = pts
    return points


def simple_point_render(pts, img_w, img_h, fov, eye, front, up):
    """
    pts: [N, 3]
    """
    canvas = np.ones((img_h, img_w, 3))

    pts = as_homogeneous(pts)

    E = compute_extrinsics(eye, front, up)
    world_2_ndc = compute_proj_to_normalized(img_w / img_h, fov)
    ndc_to_img = compute_normalized_to_img_trans(img_w / img_h, img_h)

    pts = pts @ E.T
    pts = pts @ world_2_ndc.T
    pts = homogenize(pts)

    # now filter out outliers beyond [-1, 1]
    outlier_mask = (np.abs(pts) > 1.0).any(axis=1)
    pts = pts[~outlier_mask]

    pts = pts @ ndc_to_img.T

    # now draw each point
    pts = np.rint(pts).astype(np.int32)
    xs, ys, _ = pts.T
    canvas[ys, xs] = (1, 0, 0)

    return canvas
