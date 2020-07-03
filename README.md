# GradCAM-Plug-and-Play

This is the repository for pytorch Implementation of "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". 



### Features

+ Plug-and-play usage
+ Support multiple and arbitrary layers by their full name
+ Save all useful intermediate results
+ Support various modes by specifying  `retain_graph`,  `create_graph` and `visualize` .



### Requirements

+ Python >= 3.6
+ imgaug
+ numpy
+ opencv-python
+ pytorch



### Get Started

#### Calculate Grad-CAM heatmap

First import from `gradcam.py` and create an instance:

```python
from gradcam import GradCAM
visualizer = GradCAM(model, layer_names, detach=True)
```

Each string in `layer_names` is the full name of a target layer in your model. Here is some examples for ResNet:

+ `layer4`
+ `layer4.1`
+ `layer4.1.bn1`

After forward propagation of model, the graph is established, and you can calculate Grad-CAM heatmap by:

```python
visualizer.cal_grad(y, target_class)
visualizer.cal_cam(visualize=True)
```

To show Grad-CAM heatmap on original input images:

```python
visualizer.show_cam_on_image(input) # input: (B, H, W, 3)
```

Now useful information are stored in `visualizer.vis_info` now. For each target layer name `layer_name`: 

+ `visualizer.vis_info[layer_name]['output']` :  output feature maps
+ `visualizer.vis_info[layer_name]['grad']` : gradients
+ `visualizer.vis_info[layer_name]['cam']` :  class activation maps
+ `visualizer.vis_info[layer_name]['mask_img']` :  images transformed from class activation maps
+ `visualizer.vis_info[layer_name]['heatmap_img']` : heatmap images in `cv2.COLORMAP_JET` mode
+ `visualizer.vis_info[layer_name]['vis_img']` :  visualization images

Finally, do not forget to reset state machine before next visualization:

```python
visualizer.reset_info()
```

#### Retain the graph

Sometimes we want to retain the graph after Grad-CAM calculation and do back propagation afterwards. To achieve this, specify `retain_graph=True`:

```python
visualizer.cal_grad(y, target_class, retain_graph=True)
```

#### Class activation maps for other purposes

If your purpose of calculating CAM is not for visualization, you can specify `visualize=False`:

```python
visualizer.cal_cam(visualize=False)
```

Now `visualizer.vis_info[layer_name]['cam']` is an instance of `torch.Tensor` instead of `numpy.ndarray`.

#### Differentiable class activation maps

Sometimes we want to obtain differentiable class activation maps. To achieve this, specify `detach=False` when constructing the worker:

```python
visualizer = GradCAM(model, layer_names, detach=False)
```

and specify `visualize=False` ,`retain_graph=True` and `create_graph=True`:

```python
visualizer.cal_grad(y, target_class, retain_graph=True, create_graph=True)
visualizer.cal_cam(visualize=False)
```

