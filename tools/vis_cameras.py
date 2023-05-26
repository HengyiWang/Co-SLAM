'''
camera extrinsics visualization tools
modified from https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
'''

import numpy as np
import cv2 as cv
import open3d as o3d


def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(t)

    return M_inv


def draw_cuboid(bound):
    x_min, x_max = bound[0, 0], bound[0, 1]
    y_min, y_max = bound[1, 0], bound[1, 1]
    z_min, z_max = bound[2, 0], bound[2, 1]
    points = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
              [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[0, 1, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_camera(cam_width, cam_height, f, extrinsic, color, show_axis=True):
    """
    :param extrinsic: c2w tranformation
    :return:
    """
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(extrinsic)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(extrinsic)
        return [line_set, axis]
    else:
        return [line_set]


def visualize(extrinsics=None, things_to_draw=[]):

    ########################    plot params     ########################
    cam_width = 0.64/2     # Width/2 of the displayed camera.
    cam_height = 0.48/2    # Height/2 of the displayed camera.
    focal_len = 0.20     # focal length of the displayed camera.

    ########################    original code    ########################
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if extrinsics is not None:
        for c in range(extrinsics.shape[0]):
            c2w = extrinsics[c, ...]
            camera = draw_camera(cam_width, cam_height, focal_len, c2w, color=[1, 0, 0])
            for geom in camera:
                vis.add_geometry(geom)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis)
    for geom in  things_to_draw:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()