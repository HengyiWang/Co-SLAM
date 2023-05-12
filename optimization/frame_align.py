import torch
import torch.nn as nn
import torch.nn.functional as F

class PerFrameAlignment(nn.Module):
    def __init__(self, num_frames):
        super(PerFrameAlignment, self).__init__()
        
        self.num_frames = num_frames

        self.data = nn.Parameter(
            data=torch.cat([
                torch.ones([num_frames, 2], dtype=torch.float32), 
                torch.zeros([num_frames, 2], dtype=torch.float32)
            ], -1)
        )

    def forward(self, ids):
        return self.data[ids]