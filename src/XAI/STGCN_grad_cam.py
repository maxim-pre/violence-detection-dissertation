import torch
import torch.nn.functional as F

class STGCNGradCam:
    def __init__(self, model, target_layer, normalisation_mode="per_video"):
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
        #input_tensor: [1, 3, 150, 17, 4]

        B, C, T, V, M = input_tensor.shape

        if B != 1:
            raise ValueError("Batch size must be 1")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        score = logits[0, target_class]
        score.backward()

        # shape: [4, 256, 38, 17]
        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape: [4, 256, 1, 1]
        cam = (weights * activations).sum(dim=1)  # shape: [4, 38, 17]
        cam = F.relu(cam)
        cam = cam.unsqueeze(1) # shape [4, 1, 38, 17]

        # upsample to original T (shape [4, 1, 150, 17])
        cam = F.interpolate(cam, size=(T, V), mode="bilinear", align_corners=False)

        cam = cam.squeeze(1) # shape [4, 150, 17]
        cam = cam.permute(1, 2, 0) # shape [150, 17, 4]

        cam = self._normalise(cam)
        return cam.detach()



        
