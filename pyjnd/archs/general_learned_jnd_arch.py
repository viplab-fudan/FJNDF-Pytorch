import torch
from torch import nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np

from pyjnd.models import build_network
from pyjnd.losses import build_loss
from pyjnd.metrics import calculate_metric
from pyjnd.utils import get_root_logger, imwrite, tensor2img
from pyjnd.utils.registry import ARCH_REGISTRY
from .base_learned_jnd_arch import BaseLRModel


@ARCH_REGISTRY.register()
class GeneralLRJNDModel(BaseLRModel):
    """General module to train an JND network."""

    def __init__(self, opt):
        super(GeneralLRJNDModel, self).__init__(opt)

        # define network
        self.precision = self.opt.get('network', {}).get('precision', 'fp32')
        self.net = build_network(opt['network'])

        if self.precision == 'fp16':
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.net = self.model_to_device(self.net)
        self.print_network(self.net)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        self.net_best = build_network(self.opt['network']).to(self.device)

        # define fidelity losses, such as l1 loss, mse loss
        f_opt = train_opt.get('fidelity_loss_opt')
        if f_opt:
            if isinstance(f_opt, dict):
                f_opt = [f_opt]
            losses = []
            for opt in f_opt:
                losses.append(build_loss(opt).to(self.device))
            self.cri_fidelity = nn.ModuleList(losses)
        else:
            self.cri_fidelity = None

        # define rate losses, such as si loss, si contrast loss
        f_opt = train_opt.get('constraint_loss_opt')
        if f_opt:
            if isinstance(f_opt, dict):
                f_opt = [f_opt]
            losses = []
            for opt in f_opt:
                losses.append(build_loss(opt).to(self.device))
            self.cri_cons = nn.ModuleList(losses)
        else:
            self.cri_cons = None

        # define perceptual losses, such as vgg feature loss, lpips loss
        f_opt = train_opt.get('perceptual_loss_opt')
        if f_opt:
            if isinstance(f_opt, dict):
                f_opt = [f_opt]
            losses = []
            for opt in f_opt:
                losses.append(build_loss(opt).to(self.device))
            self.cri_perceptual = nn.ModuleList(losses)
        else:
            self.cri_perceptual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']

        param_dict = {k: v for k, v in self.net.named_parameters()}
        param_keys = list(param_dict.keys())
        # set different lr for different modules if needed, e.g., lr_backbone, lr_head
        lr_keys = [i for i in optim_opt.keys() if i.startswith('lr_')]

        optim_params = []
        for key in lr_keys:
            if key.startswith('lr_'):
                module_key = key.replace('lr_', '')
                logger = get_root_logger()
                logger.info(f'Set optimizer for {module_key} with lr: {optim_opt[key]}, weight_decay: {optim_opt.get(f"weight_decay_{module_key}", 0.)}')

                optim_params.append({
                    'params': [param_dict[k] for k in param_keys if module_key in k and param_dict[k].requires_grad],
                    'lr': optim_opt.pop(key, 0.),
                    'weight_decay': optim_opt.pop(f'weight_decay_{module_key}', 0.),
                })

                # should use param_keys[:] to avoid iteration error
                for k in param_keys[:]:
                    if module_key in k:
                        param_keys.remove(k)
        
        # append the rest of the parameters
        optim_params.append({
            'params': [param_dict[k] for k in param_keys if param_dict[k].requires_grad],
        })

        # log params that will not be optimized
        for k, v in param_dict.items():
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        # remove blank param list
        for k in optim_params:
            if len(k['params']) == 0:
                optim_params.remove(k)

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)
    
    def feed_data(self, data):
        self.img_input = data['img'].to(self.device)

        # default use supervised training
        self.supervised = True
        if 'supervised' in self.opt['train']:
            self.supervised = self.opt['train']['supervised']
        
        if self.supervised:
            if 'ref_img' in data:
                self.ref_input = data['ref_img'].to(self.device)
            else:
                raise ValueError(f"Supervised traning strategies requires reference images as ground truth!")
        else:
            self.ref_input = self.img_input

        # if 'use_ref' in self.opt['train']:
        #     self.use_ref = self.opt['train']['use_ref']
        self.use_ref = False

    def net_forward(self, net):
        # current version only support origin image as input
        # TODO: support another data input, but it must can't be self.ref_input, 
        # TODO: as ref_input is used as ground truth under supervised training
        with torch.cuda.amp.autocast(enabled=(self.precision == 'fp16')):
            if self.use_ref:
                return net(self.img_input, self.ref_input)
            else:
                return net(self.img_input)

    def softclip01(self, x, k: float = 2.0):
        if not torch.is_floating_point(x):
            x = x.float()
        k = torch.as_tensor(k, dtype=x.dtype, device=x.device)
        return 0.5 * (torch.tanh(k * (x - 0.5)) + 1.0)

    def optimize_parameters(self, current_iter):
        return self._optimize_image(current_iter)

    def _optimize_image(self, current_iter):
        self.optimizer.zero_grad()
    
        with torch.cuda.amp.autocast(enabled=(self.precision == 'fp16')):
            self.output = self.net_forward(self.net)

            if self.opt['network']['only_train_y']:
                pred = self.output[:, 0:1, :, :]
                ref = self.ref_input[:, 0:1, :, :]
                ori = self.img_input[:, 0:1, :, :]
            else:
                pred = self.output
                ref = self.ref_input
                ori = self.img_input

            l_total = 0
            loss_dict = OrderedDict()
        
            # pixel loss
            if self.cri_fidelity:
                l_fidelity = 0
                for loss_fn in self.cri_fidelity:
                    l_fidelity += loss_fn(pred, ref)
                l_total += l_fidelity
                loss_dict['l_fidelity'] = l_fidelity

            if self.cri_cons:
                l_rate = 0
                for loss_fn in self.cri_cons:
                    l_rate += loss_fn(pred, ori)
                l_total += l_rate
                loss_dict['l_rate'] = l_rate

            if self.cri_perceptual:
                pred_sc = self.softclip01(pred)
                l_perceptual = 0
                for loss_fn in self.cri_perceptual:
                    l_perceptual += loss_fn(pred_sc, ref)
                l_total += l_perceptual
                loss_dict['l_perceptual'] = l_perceptual

        if self.precision == 'fp16':
            self.scaler.scale(l_total).backward()
            # self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net_forward(self.net)
        self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        # initialize best_val_loss on first call (for best-loss tracking)
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')

        use_pbar = self.opt['val'].get('pbar', False)

        # metric bookkeeping
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero current metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        pred, ori_img, ref_img = [], [], []
        for idx, val_data in enumerate(dataloader):
            img_name = osp.basename(val_data['img_path'][0])
            self.feed_data(val_data)
            self.test()
            pred.append(self.output)
            ori_img.append(self.img_input)
            ref_img.append(self.ref_input)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name:>20}')
        if use_pbar:
            pbar.close()

        if self.train_target == 'image':
            mean_val_loss = None
            if with_metrics:
                per_image_results = {name: [] for name in self.opt['val']['metrics'].keys()}

                for p, g in zip(pred, ref_img):
                    p_img = p.squeeze(0).cpu().numpy() if isinstance(p, torch.Tensor) else p
                    g_img = g.squeeze(0).cpu().numpy() if isinstance(g, torch.Tensor) else g

                    p_img = np.clip(p_img * 255.0, 0, 255).astype(np.uint8)
                    g_img = np.clip(g_img * 255.0, 0, 255).astype(np.uint8)

                    for name, opt_ in self.opt['val']['metrics'].items():
                        if self.opt['network']['only_train_y']:
                            val = calculate_metric([p_img[0:1, :, :], g_img[0:1, :, :]], opt_)
                        else:
                            val = calculate_metric([p_img, g_img], opt_)
                        per_image_results[name].append(val)

                for name, vals in per_image_results.items():
                    self.metric_results[name] = float(sum(vals) / len(vals))

                val_loss_total, val_batches = 0.0, 0
                for out, ref, ori in zip(pred, ref_img, ori_img):
                    batch_loss = 0.0
                    if self.opt['network']['only_train_y']:
                        out = out[:, 0:1, :, :]
                        ref = ref[:, 0:1, :, :]
                        ori = ori[:, 0:1, :, :]

                    if self.cri_fidelity:
                        for fn in self.cri_fidelity:
                            batch_loss += fn(out, ref).item()
                    if self.cri_cons:
                        for fn in self.cri_cons:
                            batch_loss += fn(out, ori).item()
                    if self.cri_perceptual:
                        out_sc = self.softclip01(out)
                        for fn in self.cri_perceptual:
                            batch_loss += fn(out_sc, ref).item()

                    val_loss_total += batch_loss
                    val_batches += 1

                if val_batches > 0:
                    mean_val_loss = val_loss_total / val_batches

            else:
                val_loss_total, val_batches = 0.0, 0
                for out, ref, ori in zip(pred, ref_img, ori_img):
                    batch_loss = 0.0
                    if self.opt['network']['only_train_y']:
                        out = out[:, 0:1, :, :]
                        ref = ref[:, 0:1, :, :]
                        ori = ori[:, 0:1, :, :]

                    if self.cri_fidelity:
                        for fn in self.cri_fidelity:
                            batch_loss += fn(out, ref).item()
                    if self.cri_cons:
                        for fn in self.cri_cons:
                            batch_loss += fn(out, ori).item()
                    if self.cri_perceptual:
                        out_sc = self.softclip01(out)
                        for fn in self.cri_perceptual:
                            batch_loss += fn(out_sc, ref).item()

                    val_loss_total += batch_loss
                    val_batches += 1

                if val_batches > 0:
                    mean_val_loss = val_loss_total / val_batches

            if with_metrics:
                if self.key_metric is not None:
                    # If the best metric is updated, update and save best model
                    to_update = self._update_best_metric_result(dataset_name,
                                                                self.key_metric,
                                                                self.metric_results[self.key_metric],
                                                                current_iter)
                    if to_update:
                        for name, opt_ in self.opt['val']['metrics'].items():
                            self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                        self.copy_model(self.net, self.net_best)
                        self.save_network(self.net_best, 'net_best')
                else:
                    # update each metric separately
                    for name, opt_ in self.opt['val']['metrics'].items():
                        updated = self._update_best_metric_result(dataset_name, name,
                                                                  self.metric_results[name], current_iter)
                        if updated:
                            self.copy_model(self.net, self.net_best)
                            self.save_network(self.net_best, f'net_best_{name}')

                    # save the minimal loss model as net_best_loss
                    if mean_val_loss is not None and mean_val_loss < self.best_val_loss:
                        self.best_val_loss = mean_val_loss
                        self.copy_model(self.net, self.net_best)
                        self.save_network(self.net_best, 'net_best_loss')
                        get_root_logger().info(
                            f'New best val loss: {mean_val_loss:.6f} @ iter {current_iter}, saved net_best_loss.'
                        )
            else:
                if mean_val_loss is not None and mean_val_loss < self.best_val_loss:
                    self.best_val_loss = mean_val_loss
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')
                    get_root_logger().info(
                        f'New best val loss: {mean_val_loss:.6f} @ iter {current_iter}, saved net_best.'
                    )

        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'val_metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter, save_net_label='net'):
        self.save_network(self.net, save_net_label, current_iter)
        self.save_training_state(epoch, current_iter)
