# Use dataset object

import torch
import time
import os
import random

class Mapper():
    def __init__(self, config, SLAM) -> None:
        self.config = config
        self.slam = SLAM
        self.model = SLAM.model
        self.tracking_idx = SLAM.tracking_idx
        self.mapping_idx = SLAM.mapping_idx
        self.mapping_first_frame = SLAM.mapping_first_frame
        self.keyframe = SLAM.keyframeDatabase
        self.map_optimizer = SLAM.map_optimizer
        self.device = SLAM.device
        self.dataset = SLAM.dataset

        self.est_c2w_data = SLAM.est_c2w_data
        self.est_c2w_data_rel = SLAM.est_c2w_data_rel
    
    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print('First frame mapping...')
        if batch['frame_id'] != 0:
            raise ValueError('First frame mapping must be the first frame!')
        c2w = batch['c2w'].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            rays_d_cam = batch['direction'][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'][indice_h, indice_w].to(self.device).unsqueeze(-1)


            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.keyframe.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        #     self.slam.save_mesh(0)
        
        print('First frame mapping done')
        self.mapping_first_frame[0] = 1
        return ret, loss

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframe.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(poses[1:])
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframe.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)


            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.slam.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)
            
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.slam.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
    
    def convert_relative_pose(self, idx):
        poses = {}
        for i in range(len(self.est_c2w_data[:idx])):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def run(self):

        # Start mapping
        while self.tracking_idx[0]< len(self.dataset)-1:
            if self.tracking_idx[0] == 0 and self.mapping_first_frame[0] == 0:
                batch = self.dataset[0]
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                time.sleep(0.1)
            else:
                while self.tracking_idx[0] <= self.mapping_idx[0] + self.config['mapping']['map_every']:
                    time.sleep(0.4)
                current_map_id = int(self.mapping_idx[0] + self.config['mapping']['map_every'])
                batch = self.dataset[current_map_id]
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v[None, ...]
                    else:
                        batch[k] = torch.tensor([v])
                self.global_BA(batch, current_map_id)
                self.mapping_idx[0] = current_map_id
            
                if self.mapping_idx[0] % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframe.add_keyframe(batch)
            
                if self.mapping_idx[0] % self.config['mesh']['vis']==0:
                    idx = int(self.mapping_idx[0])
                    self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose(idx)
                    self.slam.pose_eval_func()(self.slam.pose_gt, self.est_c2w_data[:idx], 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx)
                    self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='pose_r', name='output_relative.txt')
                
                time.sleep(0.2)

        idx = int(self.tracking_idx[0])       
        self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_final'])
        pose_relative = self.convert_relative_pose(idx)
        self.slam.pose_eval_func()(self.slam.pose_gt, self.est_c2w_data[:idx], 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx)
        self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='pose_r', name='output_relative.txt')

        
        
        
