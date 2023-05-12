import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad)
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

class PoseArray(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(7)) for i in range(num_frames)])


    def add_params(self, c2w, frame_id):
        with torch.no_grad(): 
            self.params[frame_id].copy_(self.get_tensor_from_camera(c2w))
        #self.params[frame_id].data = self.get_tensor_from_camera(c2w)
        

        if torch.sum(torch.isnan(self.params[frame_id].data))>0:
            print('get_tensor_from_camera warning')

        return self.params[frame_id]
    
    def get_transformation(self, id, homo=False):
        tensor = self.params[id]

        if torch.sum(torch.isnan(tensor))>0:
            print('param warning!!!!')

        if not homo:
            return self.get_camera_from_tensor(tensor)
        
        RT = self.get_camera_from_tensor(tensor)
        row = torch.tensor([[0, 0, 0, 1]]).to(RT)
        return torch.cat([RT, row], dim=0)
            

    def get_tensor_from_camera(self, RT, Tquad=False):
        """
        Convert transformation matrix to quaternion and translation.

        """
        device = RT.device
        if type(RT) == torch.Tensor:
            if RT.get_device() != -1:
                RT = RT.detach().cpu()
                gpu_id = RT.get_device()
            RT = RT.numpy()
        R, T = RT[:3, :3], RT[:3, 3]
        from mathutils import Matrix
        rot = Matrix(R)
        quad = rot.to_quaternion()
        if Tquad:
            tensor = np.concatenate([T, quad], 0)
        else:
            tensor = np.concatenate([quad, T], 0)
        tensor = torch.from_numpy(tensor).float()
        
        tensor = tensor.to(device)
        return tensor

    
    def get_camera_from_tensor(self, inputs):
        """
        Convert quaternion and translation to transformation matrix.

        """
        N = len(inputs.shape)
        if N == 1:
            inputs = inputs.unsqueeze(0)
        quad, T = inputs[:, :4], inputs[:, 4:]
        R = quad2rotation(quad)
        RT = torch.cat([R, T[:, :, None]], 2)
        if N == 1:
            RT = RT[0]
        return RT
