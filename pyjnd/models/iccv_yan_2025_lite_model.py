import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyjnd.utils.registry import MODEL_REGISTRY

# ========================================================================================
# Helper Modules
# ========================================================================================

class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        ) 
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        
    def forward(self, inp):   
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3),
             self.conv_crossh_bn(x4),
             self.conv_crossv_bn(x5)],
            1
        )
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

        conv2_weight = self.conv2.weight
        conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
        conv2_bias = self.conv2.bias

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias

        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

        conv2_bn_weight = self.conv2.weight
        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5

        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias = self.conv2.bias * k + b
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        weight = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            0
        )
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias_ = torch.cat(
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            0
        )
        bias = torch.matmul(weight_compress, bias_)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias

class MBRConv3(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):    
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat(
        [    x0,x1,x2,x3,
             self.conv_bn(x0),
             self.conv1_bn(x1),
             self.conv_crossh_bn(x2),
             self.conv_crossv_bn(x3)],
            1
        )    
        out = self.conv_out(x)
        return out

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_bias = self.conv_crossh.bias
        conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_bias = self.conv_crossv.bias
        conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

        # conv_bn
        bn = self.conv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        # conv1_bn
        bn = self.conv1_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        # conv_crossh_bn
        bn = self.conv_crossh_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        # conv_crossv_bn
        bn = self.conv_crossv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([
            conv_weight,
            conv1_weight,
            conv_crossh_weight,
            conv_crossv_weight,
            conv_bn_weight,
            conv1_bn_weight,
            conv_crossh_bn_weight,
            conv_crossv_bn_weight
        ], dim=0)

        bias = torch.cat([
            conv_bias,
            conv1_bias,
            conv_crossh_bias,
            conv_crossv_bias,
            conv_bn_bias,
            conv1_bn_bias,
            conv_crossh_bn_bias,
            conv_crossv_bn_bias
        ], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias

        return weight, bias

class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp): 
        x0 = self.conv(inp)  
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)

        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias

class FST(nn.Module):
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))  

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias

class FSTS(nn.Module):
    def __init__(self, block1, channels):
        super(FSTS, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        
    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias

class DropBlock(nn.Module):
    def __init__(self, block_size, p=0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p / block_size / block_size

    def forward(self, x):
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)

class _MobileIELLENet_Lite_Slim(nn.Module):
    def __init__(self, in_channels, out_channels, head_channels, body_channels, tail_channels):
        super(_MobileIELLENet_Lite_Slim, self).__init__()
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, 1, 1),
                nn.PReLU(head_channels),
                nn.Conv2d(head_channels, body_channels, 3, 1, 1)
            ), body_channels
        )
        self.body = FSTS(nn.Conv2d(body_channels, tail_channels, 3, 1, 1), tail_channels)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(tail_channels, tail_channels, 1), nn.Sigmoid()
        )
        self.att1 = nn.Sequential(nn.Conv2d(1, tail_channels, 1), nn.Sigmoid())
        self.tail = nn.Conv2d(tail_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        return self.tail(x4)

class _MobileIELLENet_Lite_Train(nn.Module):
    def __init__(self, in_channels, out_channels, head_channels, body_channels, tail_channels, rep_scale):
        super(_MobileIELLENet_Lite_Train, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.head_channels, self.body_channels, self.tail_channels = head_channels, body_channels, tail_channels
        self.rep_scale = rep_scale

        self.head = FST(
            nn.Sequential(
                MBRConv3(in_channels, head_channels, rep_scale=rep_scale),
                nn.PReLU(head_channels),
                MBRConv3(head_channels, body_channels, rep_scale=rep_scale)
            ), body_channels
        )
        self.body = FST(MBRConv3(body_channels, tail_channels, rep_scale=rep_scale), tail_channels)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(tail_channels, tail_channels, rep_scale=rep_scale), nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
            MBRConv1(1, tail_channels, rep_scale=rep_scale), nn.Sigmoid()
        )
        self.tail = MBRConv3(tail_channels, out_channels, rep_scale=rep_scale)
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)      
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        return self.tail(x4)

    def slim(self):
        """Converts the training network to its slim inference-time equivalent."""
        net_slim = _MobileIELLENet_Lite_Slim(self.in_channels, self.out_channels, self.head_channels, self.body_channels, self.tail_channels)
        slim_state_dict = net_slim.state_dict()
        
        for name, mod in self.named_modules():
            if isinstance(mod, (MBRConv5, MBRConv3, MBRConv1)):
                if f'{name}.weight' in slim_state_dict:
                    w, b = mod.slim()
                    slim_state_dict[f'{name}.weight'] = w
                    slim_state_dict[f'{name}.bias'] = b
            elif isinstance(mod, FST):
                if f'{name}.bias' in slim_state_dict:
                    slim_state_dict[f'{name}.bias'] = mod.bias.data
                    slim_state_dict[f'{name}.weight1'] = mod.weight1.data
                    slim_state_dict[f'{name}.weight2'] = mod.weight2.data
            elif isinstance(mod, nn.PReLU):
                 if f'{name}.weight' in slim_state_dict:
                    slim_state_dict[f'{name}.weight'] = mod.weight.data

        net_slim.load_state_dict(slim_state_dict)
        return net_slim


@MODEL_REGISTRY.register()
class MobileIENet_Lite(nn.Module):
    """
    MobileIE: A lightweight network for image enhancement and restoration.
    This class serves as a wrapper that internally manages a training-time version
    and a fast inference-time (slim) version of the network.

    The model automatically switches to the slim version when .eval() is called.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for YUV or RGB).
        out_channels (int): Number of output channels.
        base_channels (int): Number of channels in the main network body.
        rep_scale (int): Expansion factor used in MBRConv blocks.
        only_train_y (bool): If True, only the Y channel (the first channel) will be processed.
        effect_yml_path (str, optional): Path to the YAML file for model configuration.
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 head_channels=8,
                 body_channels=8,
                 tail_channels=12,
                 rep_scale=4,
                 only_train_y=True,
                 effect_yml_path=None,
                 precision=None
                 ):
        super(MobileIENet_Lite, self).__init__()

        if effect_yml_path and os.path.isfile(effect_yml_path):
            with open(effect_yml_path, 'r', encoding='utf-8') as f:
                opt = yaml.safe_load(f)
            in_channels = opt.get('in_channels', in_channels)
            out_channels = opt.get('out_channels', out_channels)
            head_channels = opt.get('head_channels', head_channels)
            body_channels = opt.get('body_channels', body_channels)
            tail_channels = opt.get('tail_channels', tail_channels) # 新增
            rep_scale = opt.get('rep_scale', rep_scale)
            only_train_y = opt.get('only_train_y', only_train_y)

        self.only_train_y = only_train_y
        
        net_in_channels = 1 if only_train_y else in_channels
        net_out_channels = 1 if only_train_y else out_channels

        self.network = _MobileIELLENet_Lite_Train(
            in_channels=net_in_channels,
            out_channels=net_out_channels,
            head_channels=head_channels,
            body_channels=body_channels,
            tail_channels=tail_channels, # 新增
            rep_scale=rep_scale
        )
        
        self._slim_network = None
        self._is_slim = False

    def train(self, mode: bool = True):
        """Switches the model to training mode."""
        super().train(mode)
        if mode:
            # When switching back to training, discard the slim model
            self._slim_network = None
            self._is_slim = False
        return self

    def eval(self):
        """Switches the model to evaluation mode, automatically creating the slim version."""
        super().eval()
        if not self._is_slim:
            # Create and store the slim model if it doesn't exist
            self._slim_network = self.network.slim()
            device = next(self.parameters()).device
            self._slim_network.to(device)
            self._is_slim = True
        return self
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Uses the training network in train mode and the slim network in eval mode.

        Args:
            x: Input tensor with shape (N, C, H, W).
        Returns:
            Output tensor with shape (N, C, H, W).
        """
        if self.only_train_y:
            y_in = x[:, :1]
            uv = x[:, 1:]
            net_input = y_in
        else:
            net_input = x
            
        # Add a global residual connection
        residual = net_input

        if self.training:
            # Use the full network during training
            processed = self.network(net_input)
        else:
            # Use the slim network during evaluation
            # The .eval() call ensures _slim_network is created
            if self._slim_network is None:
                # This is a fallback in case .eval() was not called explicitly
                self._slim_network = self.network.slim()
                device = next(self.parameters()).device
                self._slim_network.to(device)
                self._is_slim = True
            processed = self._slim_network(net_input)
            
        # The original LLE network is a direct mapping, not a residual model.
        # If you need a residual model like IQNet, use `output = residual + processed`.
        # Here we follow the original MobileIE-LLE logic.
        output = processed + residual

        if self.only_train_y:
            return torch.cat([output, uv], dim=1)
        else:
            return output