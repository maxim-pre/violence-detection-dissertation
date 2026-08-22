import torch
import torch.nn.functional as F

class STGCNGradCam:
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
        heatmap: [T, V, M]
            where:
                T = num_frames
                V = num_joints
                M = num_people
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
        '''
        input_tensor: [1, C, T, V, M]

        returns:
            cam: [T, V, M]
            logits: [1, num_classes]
        '''
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

        # shape: [M, laster_layer_out_channels, reduced_time, V]
        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape: [M, laster_layer_out_channels, 1, 1]
        cam = (weights * activations).sum(dim=1)  # shape: [M, reduced_time, V]
        cam = F.relu(cam)
        cam = cam.unsqueeze(1) # shape [M, 1, reduced_time, V]

        # upsample to original T (shape [M, 1, T, V])
        cam = F.interpolate( 
            cam, 
            size=(T, V), 
            mode="bilinear", 
            align_corners=False,
        )

        cam = cam.squeeze(1) # shape [M, T, V]
        cam = cam.permute(1, 2, 0) # shape [T, V, M]

        cam = self._normalise(cam)
        return cam.detach()



        
