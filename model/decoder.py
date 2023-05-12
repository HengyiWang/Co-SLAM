# Package imports
import torch
import torch.nn as nn
import tinycudann as tcnn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))

class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])
    
    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim 
                
                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim 
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))

class ColorSDFNet(nn.Module):
    '''
    Color grid + SDF grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet, self).__init__()
        self.config = config
        self.color_net = ColorNet(config, 
                input_ch=input_ch+input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    def forward(self, embed, embed_pos, embed_color):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)
    
class ColorSDFNet_v2(nn.Module):
    '''
    No color grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v2, self).__init__()
        self.config = config
        self.color_net = ColorNet(config, 
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)