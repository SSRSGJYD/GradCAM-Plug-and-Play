from types import FunctionType
import cv2
from imgaug import augmenters as iaa
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class GradCam(object):
    def __init__(self, model, layer_names, detach=True):
        '''
        Construct the worker class.

        Args:
            model: the target model which is an instance of nn.Module, nn.DataParallel or nn.parallel.DistributedDataParallel
            layer_names: a list of layer names, each layer name is a string formatted as '(module.)*'
            detach: decide whether to detach the tensor that is to be saved from graph
        '''
        self.model = model
        self.cuda_device = self.model.device
        self.layer_names = layer_names
        self._initialize(detach)

    def _initialize(self, detach):
        '''
        Add hooks to target layers.

        Args:
            detach: decide whether to detach the tensor that is to be saved from graph
        '''
        self.vis_info = {}
        for module_name in self.layer_names:
            self.vis_info[module_name] = {}
            self.vis_info[module_name]['output'] = []
            self.vis_info[module_name]['grad'] = []
            self.vis_info[module_name]['cam'] = []
            self.vis_info[module_name]['vis_img'] = []

        self.forward_hook_handles = []
        self.backward_hook_handles = []
        self.add_hook(detach)

    def add_hook(self, detach):
        '''
        Add hooks to target layers.

        Args:
            detach: decide whether to detach the tensor that is to be saved from graph
        '''
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            self._add_hook(self.model.module, detach)
        else:
            self._add_hook(self.model, detach)
    
    def _add_hook(self, net, detach, prefix=''):
        '''
        Worker function of adding hooks to target layers.

        Args:
            net: instance of nn.Module
            detach: decide whether to detach the tensor that is to be saved from graph
            prefix: a string formatted as '(module.)*'
        '''
        if hasattr(net, '_modules'):
            for module_name, module in net._modules.items():
                new_prefix = prefix + module_name + '.'
                current_module_name = prefix + module_name
                if current_module_name in self.layer_names:
                    if detach:
                        save_output_code = compile('def save_output' + module_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + current_module_name + '\"]["output"].append(output.detach());', "<string>", "exec")
                    else:
                        save_output_code = compile('def save_output' + module_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + current_module_name + '\"]["output"].append(output);', "<string>", "exec")
                    func_space = {'self': self}
                    func_space.update(globals())
                    save_output = FunctionType(
                        save_output_code.co_consts[0], func_space, "save_output")
                    h = module.register_forward_hook(save_output)
                    self.forward_hook_handles.append(h)

                    save_gradient_code = compile(
                        'def save_gradient' + module_name +
                        '(module, input_grad, output_grad): '
                        'vis_info = getattr(self, "vis_info");'
                        'vis_info[\"' + current_module_name + '\"]["grad"].append(output_grad[0]);', "<string>", "exec")
                    save_gradient = FunctionType(
                        save_gradient_code.co_consts[0], func_space, "save_gradient")
                    h = module.register_backward_hook(save_gradient)
                    self.backward_hook_handles.append(h)
                self._add_hook(module, detach, new_prefix)

    def remove_hook(self):
        """
        Remove all hooks.
        
        """
        for h in self.forward_hook_handles:
            h.remove()
        self.forward_hook_handles = []
        for h in self.backward_hook_handles:
            h.remove()
        self.backward_hook_handles = []

    def cal_grad(self, y, target_class, retain_graph=False, create_graph=False):
        """
        Backpropagate to calculate gradients.

        Args:
            y: output of model
            target_class: target label to be visualized
            retain_graph: whether to retain the graph
            create_graph: whether to create a graph for gradients
        """
        
        one_hots = torch.zeros(y.shape[0], y.shape[1]).cuda(self.model.device)
        one_hots[:, target_class] = 1
        ys = torch.sum(one_hots * y)
        ys.backward(retain_graph=retain_graph, create_graph=create_graph)

    def cal_cam(self, visualize=True):
        '''
        Display GradCAM heatmaps on original images.
        
        Args:
            imgs: original images as a tensor shaped (B, H, W, 3)
        '''
        self._cat_info()
        for key in self.vis_info.keys():
            grads_val = self.vis_info[key]['grad'] # (B, C, H, W)
            feature = self.vis_info[key]['output'] # (B, C, H, W)
            weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)
            cam = weights * feature
            cam = torch.sum(cam, dim=1)
            cam = F.relu(cam) # (B, H, W)
            # normalize to (0, 1)
            tmp = cam.view(cam.shape[0], -1)
            max_value = torch.max(tmp, dim=1)[0] # (batch)
            max_value = max_value.unsqueeze(dim=1).unsqueeze(dim=1) # (batch, 1, 1)
            cam = torch.div(cam, max_value)
            # save to vis_info
            if visualize:
                cam = cam.cpu().numpy()
                self.vis_info[key]['cam'] = cam
            else:
                self.vis_info[key]['cam'] = cam

    def show_cam_on_image(self, imgs):
        '''
        Display GradCAM heatmap on original images.

        Args:
            imgs: original images as a tensor shaped (B, H, W, 3)
        '''
        imgs = imgs / 255.
        iaa_resize = iaa.Resize({"height": imgs.shape[1], "width": imgs.shape[2]}, interpolation="linear")
        for key in self.vis_info.keys():
            cams = self.vis_info[key]['cam']
            cams = np.transpose(cams, (1, 2, 0))
            vis_imgs = []
            heatmap_imgs = []
            mask_imgs = []
            masks = iaa_resize.augment_image(cams) * 255
            masks = np.transpose(masks, (2, 0, 1))
            masks = masks.astype('uint8')
            for i in range(imgs.shape[0]):
                img = imgs[i]  # (H, W, 3)
                heatmap = cv2.applyColorMap(masks[i], cv2.COLORMAP_JET)
                mask_imgs.append(masks[i])
                heatmap_imgs.append(heatmap)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.float32(img)
                cam = cam / np.max(cam)
                vis_img = np.uint8(255 * cam)
                vis_imgs.append(vis_img)

            self.vis_info[key]['mask_img'] = mask_imgs
            self.vis_info[key]['vis_img'] = vis_imgs
            self.vis_info[key]['heatmap_img'] = heatmap_imgs

    def reset_info(self):
        '''
        Reset vis_info dictionary to empty.
        '''
        for module_name in self.layer_names:
            self.vis_info[module_name] = {}
            self.vis_info[module_name]['output'] = []
            self.vis_info[module_name]['grad'] = []
            self.vis_info[module_name]['cam'] = []
            self.vis_info[module_name]['imgs'] = []

    def _cat_info(self):
        ''' 
        Concatenate tensor list into one tensor.
        '''
        for module_name in self.layer_names:
            if isinstance(self.vis_info[module_name]['cam'], list) and len(self.vis_info[module_name]['cam']) > 0:
                self.vis_info[module_name]['cam'] = torch.cat(
                    self.vis_info[module_name]['cam'], dim=0)
            if isinstance(self.vis_info[module_name]['output'], list) and len(self.vis_info[module_name]['output']) > 0:
                self.vis_info[module_name]['output'] = torch.cat(
                    self.vis_info[module_name]['output'], dim=0)
            if isinstance(self.vis_info[module_name]['grad'], list) and len(self.vis_info[module_name]['grad']) > 0:
                self.vis_info[module_name]['grad'] = torch.cat(
                    self.vis_info[module_name]['grad'], dim=0)