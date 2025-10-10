import torch

from collections import OrderedDict
from pyjnd.default_model_configs import DEFAULT_CONFIGS
# from pyjnd.utils.registry import ARCH_REGISTRY
from pyjnd.models import build_network
from pyjnd.utils import imread2tensor

from pyjnd.archs.arch_util import load_pretrained_network

class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            seed=123,
            check_input_range=True,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        # self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        # self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        # self.score_range = DEFAULT_CONFIGS[metric_name].get('score_range', None)
        # if self.metric_mode is None:
        #     self.metric_mode = kwargs.pop('metric_mode')
        # elif 'metric_mode' in kwargs:
        #     kwargs.pop('metric_mode')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        if metric_name == 'compare2score':
            self.as_loss=True
            self.loss_reduction='none'
        # disable input range check when used as loss
        self.check_input_range = check_input_range if not as_loss else False

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        
        self.precision = 'fp32'
        if 'precision' in kwargs:
            self.precision = kwargs.pop('precision')
        
        self.net = build_network(net_opts, precision=self.precision)
        self.net = self.net.to(self.device)
        self.net.eval()

        self.seed = seed

        self.dummy_param = torch.nn.Parameter(torch.empty(0)).to(self.device)
    
    def load_weights(self, weights_path, weight_keys='params'):
        load_pretrained_network(self.net, weights_path, weight_keys=weight_keys)
    
    def is_valid_input(self, x) -> torch.Tensor:
        if x is not None:
            assert isinstance(x, torch.Tensor), 'Input must be a torch.Tensor'
            network_type = DEFAULT_CONFIGS[self.metric_name]['metric_opts']['type']

            if network_type in ['FrequencyJNDModel', 'SpatialJNDModel', 'TopDownJNDModel']:
                if x.dim() == 4:
                    x = x.squeeze(0)
                assert x.dim() == 3, f'For {network_type!s}, Input must be 3D tensor (C, H, W)'
                assert x.shape[0] in [1, 3], 'Input must be RGB or gray image'
            else:
                if x.dim() == 3 and x.shape[0] in [1, 3]:
                    x = x.unsqueeze(0)
                assert x.dim() == 4, f'For {network_type!s}, Input must be 4D tensor (B, C, H, W)'
                assert x.shape[1] in [1, 3], 'Input must be RGB or gray image'
                if self.check_input_range:
                    assert x.min() >= 0 and x.max() <= 1, f'Input must be normalized to [0, 1], but got min={x.min():.4f}, max={x.max():.4f}'

        return x

    def forward(self, img, ref=None, **kwargs):
        device = self.dummy_param.device

        with torch.set_grad_enabled(self.as_loss):

            if not torch.is_tensor(img):
                img = imread2tensor(img, rgb=True)
                img = img.unsqueeze(0)
                if self.metric_mode == 'FR':
                    assert ref is not None, 'Please specify reference image for Full Reference metric'
                    ref = imread2tensor(ref, rgb=True)
                    ref = ref.unsqueeze(0)
                    self.is_valid_input(ref)
                
            img = self.is_valid_input(img)

            # if self.metric_mode == 'FR':
            #     assert ref is not None, 'Please specify reference image for Full Reference metric'
            #     output = self.net(img.to(device), ref.to(device), **kwargs)
            # elif self.metric_mode == 'NR':
            if self.precision == 'fp16':
                img = img.half()
            output = self.net(img.to(device), **kwargs)

        return output
