import torch 
import torch.nn.functional as F

class SmoothGradCAM:
    def __init__(self, model, target_layer, num_samples=20, noise_std=0.05, normalisation_mode="per_frame", noise_mode="per_frame"):
        self.model = model
        self.target_layer = target_layer
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.normalisation_mode = normalisation_mode
        self.noise_mode = noise_mode
        self.gradients = None
        self.activations = None
        
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)


        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

        if self.noise_mode not in ["per_frame", "per_video"]:
            raise ValueError("noise_mode must be either 'per_frame' OR 'per_video'")

    def _save_activations(self, module, inputs, output):
        self.activations = output
    
    def _save_gradients(self, module, inputs, grad_output):
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def _normalise(self, cam):
        '''
            cam shape: [32, 224, 224] or [batch_size * num_frames, 224, 224] 
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
    
    def _get_noise(self, input_tensor):

        B, T, C, H, W = input_tensor.shape

        if self.noise_mode == "per_frame":
            noise = torch.randn(
                B, T, C, H, W, 
                device=input_tensor.device, 
            ) * self.noise_std
        
        elif self.noise_mode == "per_video":
            noise = torch.randn(
                B, 1, C, H, W, 
                device=input_tensor.device, 
            ) * self.noise_std
        else:
            raise ValueError("noise_mode must be either 'per_frame' OR 'per_video'")

        return noise
    
    def _generate_single_cam(self, input_tensor, target_class):
        '''
        input_tensor shape = [1, 32, 3, 224, 224]
        '''

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        return cam
    

    def generate_heatmap(self, input_tensor, target_class=None):
        B, T, C, H, W = input_tensor.shape

        if B != 1:
            raise ValueError("Batch size must be 1")

        # put model in eval mode (disable dropout etc..)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.detach()
        logits = self.model(input_tensor)

        if target_class is None:
            # pick the class with the highest score if no target class is provided
            target_class = logits.argmax(dim=1).item()
        
        cam_accumulator = None 

        for i in range(self.num_samples):
            noise = self._get_noise(input_tensor)
            noisy_input_tensor = input_tensor + noise 
            noisy_input_tensor = noisy_input_tensor.detach().clone().requires_grad_(True)

            cam = self._generate_single_cam(noisy_input_tensor, target_class)

            if cam_accumulator is None:
                cam_accumulator = torch.zeros_like(cam)
            
            cam_accumulator += cam.detach()
            print(f"processed {i+1}/{self.num_samples}")
        
        # [32, 7, 7]
        smooth_cam = cam_accumulator / self.num_samples

        # [32, 1, 7, 7] => need this shape for interpolation
        smooth_cam = smooth_cam.unsqueeze(1)

        # resize heatmaps to the original input size: [B, num_frames, H, W]
        smooth_cam = F.interpolate(
            smooth_cam,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # [32, 224, 224]
        smooth_cam = smooth_cam.squeeze(1)
        smooth_cam = self._normalise(smooth_cam)
        smooth_cam = smooth_cam.view(B, T, H, W)

        return smooth_cam.detach(), logits.detach()