# Co-SLAM Documentation

### [Paper](https://arxiv.org/pdf/2304.14377.pdf) | [Project Page](https://hengyiwang.github.io/projects/CoSLAM) | [Video](https://hengyiwang.github.io/projects/Co-SLAM/videos/presentation.mp4)

> Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM <br />
> [Hengyi Wang](https://hengyiwang.github.io/), [Jingwen Wang](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<br />
> CVPR 2023

This is the documentation for Co-SLAM that contains data capture, details of different hyper-parameters.



## Create your own dataset using iphone/ipad pro

1. Download [strayscanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) in App Store
2. Record the RGB-D video
3. The output fold should be at `Files/On My iPad/Stray Scanner/`
4. Create your own config files for dataset, set `dataset: 'iphone'` , check `./configs/iPhone` for details. (Note intrinsics given by Stray Scanner should be divided by 7.5 to align the RGB-D frame)
5. Create your config file for specific scene, you can define the scene bound yourself, or use provided `vis_bound.ipynb` to determine scene bound

NOTE: The resolution of depth map is (256, 192), which is a bit too small. The camera tracking of neural SLAM won't be very robust on the iphone dataset. It is recommended to use RealSense for data capturing. Any suggestions on this would be welcome.

## Parameters

### Tracking

```python
iter: 10 # num of iterations for tracking
sample: 1024 # num of samples for tracking
pc_samples: 40960 # num of samples for tracking using pc loss (Not used)
lr_rot: 0.001 # lr for rotation
lr_trans: 0.001 # lr for translation
ignore_edge_W: 20 # ignore the edge of image (W) 
ignore_edge_H: 20 # ignore the edge of image (H)
iter_point: 0 # num of iterations for tracking using pc loss (Not used)
wait_iters: 100 # Stop optimizing if no improvement for k iterations 
const_speed: True # Constant speed assumption for initializing pose
best: True # Use the pose with smallest loss/Use last pose
```



### Mapping

```python
sample: 2048 # Number of pixels used for BA
first_mesh: True # Save first mesh
iters: 10 # num of iterations for BA
  
# lr for representation, an interesting observation is that
# if you set lr_embed=0.001, lr_decoder=0.01, this makes
# decoder relies more on coordinate encoding, results in
# better completion. This is suitable for room-scale scene, but
# is not suitable for TUM RGB-D...
lr_embed: 0.01 # lr for HashGrid
lr_decoder: 0.01 # lr for decoder
  
lr_rot: 0.001 # lr for rotation
lr_trans: 0.001 # lr for translation
keyframe_every: 5 # Select keyframe every 5 frames
map_every: 5 # Perform BA every 5 frames
n_pixels: 0.05 # num of pixels saved for each frame
first_iters: 500 # num of iterations for first frame mapping
 
# As we perform global BA, we need to make sure 1) for every 
# iteration, there should be samples from current frame, and
# 2) Do not sample too many pixels on current frame, which may 
# introduce bias, it is suggested min_pixels_cur = 0.01 * #samples
optim_cur: False # For challenging scenes, avoid optimizing current frame pose during BA
min_pixels_cur: 20 # min pixels sampled for current frame
map_accum_step: 1 # num of steps for accumulating gradient for model
pose_accum_step: 5 # num of steps for accumulating gradient for pose
map_wait_step: 0 # wait n iterations to start update model
filter_depth: False # Filter out outliers or not
```



### Parametric encoding

```python
enc: 'HashGrid' # Type of grid, including 'DenseGrid, TiledGrid' as describled in tinycudann
tcnn_encoding: True # Use tcnn encoding
hash_size: 19 # Hash table size, refer to our paper for different settings of each dataset
voxel_color: 0.08 # Voxel size for color grid (if applicable)
voxel_sdf: 0.04 # Voxel size for sdf grid (Larger than 10 means voxel dim instead, i.e. fixed resolution)
oneGrid: True # Use only OneGrid
```



### Coordinate encoding

```python
enc: 'OneBlob' # Type of coordinate encoding
n_bins: 16 # Number of bins for OneBlob
```



### Decoder

```python
geo_feat_dim: 15 # Dim of geometric feature for color decoder
hidden_dim: 32 # Dim of SDF MLPs
num_layers: 2 # Num of layers for SDF MLP
num_layers_color: 2 # Num of layers for color MLPs
hidden_dim_color: 32 # Dim of color MLPs
tcnn_network: False # Use tinycudann MLP or Pytorch MLP
```



### Training

```python
rgb_weight: 5.0 # weight of rgb loss
depth_weight: 0.1 # weight of depth loss
sdf_weight: 1000 # weight of sdf loss (truncation)
fs_weight: 10 # weight of sdf loss (free space)
eikonal_weight: 0 # weight of eikonal (Not used)
smooth_weight: 0.001 # weight of smoothness loss (Smaller, as it is applied for feature)
smooth_pts: 64 # Dim of random sampled grid for smoothness
smooth_vox: 0.1 # Voxel size of random sampled grid for smoothness
smooth_margin: 0.05 # Margin of sampled grid
#n_samples: 256
n_samples_d: 96 # Num of sample points for rendering
range_d: 0.25 # Sampled range for depth-guided sampling [-25cm, 25cm]
n_range_d: 21 # Num of depth-guided sample points
n_importance: 0 # Num of sample points for importance sampling
perturb: 1 # Random perturbation (1:True)
white_bkgd: False
trunc: 0.1 # truncation range (10cm for room-scale scene, 5cm for RUM RGBD)
rot_rep: 'quat' # Rotation representation. (Axis angle does not support identity init)
```

