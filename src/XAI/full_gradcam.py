import torch 
import torch.nn.functional as F

#---------
# THIS IS ACTUALLY MULTILAYER GRADCAM, BUT I NAMED IT FULLGRADCAM
#---------
class FullGradCAM:
    def __init__(self, model, target_layers, normalisation_mode="per_video"):
        self.model = model
        self.target_layers = target_layers 
        self.normalisation_mode = normalisation_mode

        self.gradients = {}
        self.activations = {}

        self.forward_hooks = []
        self.backward_hooks = []

        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

        for i, layer in enumerate(target_layers):
            forward_hook = layer.register_forward_hook(self._make_activation_hook(i))
            backward_hook = layer.register_full_backward_hook(self._make_gradient_hook(i))

            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)


    def _make_activation_hook(self, layer_index):
        def save_activation(module, inputs, output):
            self.activations[layer_index] = output
        
        return save_activation

    def _make_gradient_hook(self, layer_index):
        def save_gradient(module, inputs, output):
            self.gradients[layer_index] = output[0]
        
        return save_gradient
    
    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        
        for hook in self.backward_hooks:
            hook.remove()
    
    def _normalise(self, cam):
        if self.normalisation_mode == "per_frame":
            cam_min = cam.flatten(1).min(dim=1).values.view(-1, 1, 1)
            cam_max = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
        elif self.normalisation_mode == "per_video":
            cam_min = cam.min()
            cam_max = cam.max()
        else:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def generate_heatmap(self, input_tensor, target_class=None):

        B, T, C, H, W = input_tensor.shape
        
        if B != 1:
            raise ValueError("Batch size must be 1")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        self.activations.clear()
        self.gradients.clear()

        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1) # pick predicted class

        score = logits[0, target_class]
        score.backward()

        layer_cams = []

        for layer_index in range(len(self.target_layers)):
            activation = self.activations[layer_index] 
            gradient = self.gradients[layer_index]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            layer_cam = (weights * activation).sum(dim=1)
            layer_cam = F.relu(layer_cam)
            layer_cam = layer_cam.unsqueeze(1)

            layer_cam = F.interpolate(
                layer_cam,
                size=(H,W),
                mode="bilinear",
                align_corners=False
            )
            layer_cam = layer_cam.squeeze(1) # [32, 224, 224]
            layer_cam = layer_cam.view(B,T,H,W)
            layer_cams.append(layer_cam)
        
        stacked_cams = torch.stack(layer_cams, dim=0)

        final_cam = stacked_cams.mean(dim=0) # [1, 32, 224, 224]

        final_cam = final_cam.reshape(B*T,H,W)
        final_cam = self._normalise(final_cam)
        final_cam = final_cam.reshape(B,T,H,W)

        return final_cam.detach()


            











