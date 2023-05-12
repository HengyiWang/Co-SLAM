import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, quaternion_to_axis_angle

# TODO: Identity would cause the problem...
def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(*batch_dims,3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def at_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = axis_angle_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def qt_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = quaternion_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def six_t_to_transform_matrix(rot, trans):
    """
    :param rot: 6d rotation [bs, 6]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = rotation_6d_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return 