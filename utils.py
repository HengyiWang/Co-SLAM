import os
import torch
import numpy as np
import trimesh
import marching_cubes as mcubes
from matplotlib import pyplot as plt


#### GO-Surf ####
def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))
#### ####

def getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size=None, resolution=None):

    if not isinstance(x_max, float):
        x_max = float(x_max)
        x_min = float(x_min)
        y_max = float(y_max)
        y_min = float(y_min)
        z_max = float(z_max)
        z_min = float(z_min)
    
    if voxel_size is not None:
        Nx = round((x_max - x_min) / voxel_size + 0.0005)
        Ny = round((y_max - y_min) / voxel_size + 0.0005)
        Nz = round((z_max - z_min) / voxel_size + 0.0005)

        tx = torch.linspace(x_min, x_max, Nx + 1)
        ty = torch.linspace(y_min, y_max, Ny + 1)
        tz = torch.linspace(z_min, z_max, Nz + 1)
    else:
        tx = torch.linspace(x_min, x_max, resolution)
        ty = torch.linspace(y_min, y_max,resolution)
        tz = torch.linspace(z_min, z_max, resolution)


    return tx, ty, tz

def get_batch_query_fn(query_fn, num_args=1, device=None):

    if num_args == 1:
        fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(device))
    else:
        fn = lambda f, f1, i0, i1: query_fn(f[i0:i1, None, :].to(device), f1[i0:i1, :].to(device))


    return fn

#### NeuralRGBD ####
@torch.no_grad()
def extract_mesh(query_fn, config, bounding_box, marching_cube_bound=None, color_func = None, voxel_size=None, resolution=None, isolevel=0.0, scene_name='', mesh_savepath=''):
    '''
    Extracts mesh from the scene model using marching cubes (Adapted from NeuralRGBD)
    '''
    # Query network on dense 3d grid of points
    if marching_cube_bound is None:
        marching_cube_bound = bounding_box

    x_min, y_min, z_min = marching_cube_bound[:, 0]
    x_max, y_max, z_max = marching_cube_bound[:, 1]

    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size, resolution)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)

    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    bounding_box_cpu = bounding_box.cpu()

    if config['grid']['tcnn_encoding']:
        flat = (flat - bounding_box_cpu[:, 0]) / (bounding_box_cpu[:, 1] - bounding_box_cpu[:, 0])

    fn = get_batch_query_fn(query_fn, device=bounding_box.device)

    chunk = 1024 * 64
    raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / config['data']['sc_factor'] - config['data']['translation']


    if color_func is not None and not config['mesh']['render_color']:
        if config['grid']['tcnn_encoding']:
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])


        fn_color = get_batch_query_fn(color_func, 1)

        chunk = 1024 * 64
        raw = [fn_color(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]

        sh = vert_flat.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
    
    elif color_func is not None and config['mesh']['render_color']:
        print('rendering surface color')
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        vertex_normals = torch.from_numpy(mesh.vertex_normals)
        fn_color = get_batch_query_fn(color_func, 2, device=bounding_box.device)
        raw = [fn_color(torch.from_numpy(vertices), vertex_normals,  i, i + chunk).cpu().data.numpy() for i in range(0, vertices.shape[0], chunk)]

        sh = vertex_normals.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)

    else:
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

    
    os.makedirs(os.path.split(mesh_savepath)[0], exist_ok=True)
    mesh.export(mesh_savepath)

    print('Mesh saved')
    return mesh
#### #### 

#### SimpleRecon ####
def colormap_image(
        image_1hw,
        mask_1hw=None,
        invalid_color=(0.0, 0, 0.0),
        flip=True,
        vmin=None,
        vmax=None,
        return_vminvmax=False,
        colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.
    Args:
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.
    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.
    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
        plt.cm.get_cmap(colormap)(
            torch.linspace(0, 1, 256)
        )[:, :3]
    ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
    ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw





