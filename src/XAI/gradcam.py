import torch 
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer, normalisation_mode="per_frame"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.normalisation_mode = normalisation_mode


        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)


        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

    def _save_activations(self, module, inputs, output):
        self.activations = output
    
    def _save_gradients(self, module, inputs, grad_output):
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
    
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
        # input_tensor: [B, num_frames, C, H, W]

        # put model in eval mode (disable dropout etc..)
        self.model.eval()

        self.model.zero_grad(set_to_none=True)

        input_tensor = input_tensor.detach().clone().requires_grad_(True)

        logits = self.model(input_tensor)

        batch_size = logits.size(dim=0)

        if target_class is None:
            # pick the class with the highest score if no target class is provided
            target_class = logits.argmax(dim=1)
        else:
            raise ValueError("target_class must be None. Code not implemented yet")
        
        
        # sums the logits for the predicted class across the batchs
        score = logits[
            torch.arange(batch_size, device=logits.device), 
            target_class
        ].sum()

        score.backward()

        # activation: [B * num_frames, 1280, 7, 7]
        # gradient: [B * num_frames, 1280, 7, 7]
        activation = self.activations
        gradient = self.gradients
        print(f"activation_shape: {activation.shape}")
        print(f"gradient_shape: {gradient.shape}")


        # weights: [B * num_frames, 1280, 1, 1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        print(f"weights shape: {weights.shape}")

        # cam: [B * num_frames, 7, 7]
        cam = (weights * activation).sum(dim=1)
        print(f"cam shape: {cam.shape}")

        cam = F.relu(cam)

        # Normalize the CAM to [0, 1]
        cam = self._normalise(cam)

        B, T, C, H, W = input_tensor.shape

        #convert back to: [B, num_frames, 7, 7]
        cam = cam.view(B, T, cam.shape[-2], cam.shape[-1])

        # resize heatmaps to the original input size: [B, num_frames, H, W]
        cam = F.interpolate(
            cam.view(B * T, 1, cam.shape[-2], cam.shape[-1]),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        cam = cam.view(B, T, H, W)

        return cam.detach(), logits.detach()