import torch 
import torch.nn.functional as F


class FullGradCAM:
    def __init__(self, model, target_layers, normalisation_mode="per_frame"):
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
            backward_hook = layer.register_forward_hook(self._make_gradient_hook(i))

            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)


    def _make_activation_hook(self, layer_index):
        def save_activation(module, inputs, output):
            self.activations[layer_index] = output
        
        return save_activation

    def _make_gradient_hook(self, layer_index):
        def save_gradient(module, inputs, output):
            self.gradients[layer_index] = output
        
        return save_gradient
    
    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        
        for hook in self.backward_hooks:
            hook.remove()
    
    def _normalise(self, cam):
        '''
            cam shape: [32, 7, 7] or [batch_size * num_frames, 7, 7] 
        '''
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
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        self.activations.clear()
        self.gradients.clear()

        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        logits = self.model(input_tensor)
        batch_size = logits.size(dim=0)

        if target_class is None:
            # pick the class with the highest score if no target class is provided
            target_class = logits.argmax(dim=1)
        else:
            raise ValueError("target_class must be None. Code not implemented yet")

        score = logits[
            torch.arange(batch_size, device=logits.device), 
            target_class
        ].sum()

        score.backward()

        B, T, C, H, W = input_tensor.shape
        layer_cams = []

        for layer_index in range(len(self.target_layers)):
            activation = self.activations[layer_index]
            gradient = self.gradients[layer_index]
            print(f"layer: {layer_index}")
            print(f"activation_size: {activation.shape}")
            print(f"gradient_size: {gradient.shape}")
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            layer_cam = (weights * activation).sum(dim=1)
            layer_cam = F.relu(layer_cam)

            feature_H = layer_cam.shape[-2]
            feature_W = layer_cam.shape[-1]
            layer_cam = layer_cam.view(B, T, feature_H, feature_W)
            layer_cam = F.interpolate(
                layer_cam.view(B*T,1,feature_H, feature_W),
                size=(H,W),
                mode="bilinear",
                align_corners=False
            )
            layer_cam = layer_cam.view(B,T,H,W)
            layer_cams.append(layer_cam)
        
        stacked_cams = torch.stack(layer_cams, dim=0)

        final_cam = stacked_cams.mean(dim=0) # [1, 32, 224, 224]
        print(f"final_cam_shape {final_cam.shape}")

        final_cam = final_cam.reshape(B*T,H,W)
        final_cam = self._normalise(final_cam)
        final_cam = final_cam.reshape(B,T,H,W)

        return final_cam.detach(), logits.detach(), [cam.detach() for cam in layer_cams]


            











