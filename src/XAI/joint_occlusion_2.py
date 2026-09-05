import torch
import torch.nn.functional as F



class JointOcclusion:
    def __init__(self, model, temporal_window_size=9, use_temporal_window=True, normalisation_mode="per_video"):
        self.model = model 
        self.temporal_window_size = temporal_window_size
        self.model.eval()
        self.use_temporal_window = use_temporal_window
        self.normalisation_mode=normalisation_mode

        assert self.temporal_window_size % 2 != 0

        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

    def _predict(self, input_tensor):
        # input tensor [1, 3, 150, 17, 4]

        logits = self.model(input_tensor) # [1, num_classes]
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities

    def _get_window_bounds(self, frame_index, num_frames):
        half_window = self.temporal_window_size // 2
        start = max(0, frame_index - half_window)
        end = min(num_frames, frame_index + half_window + 1)

        return start, end
    
    def _occlude_joint(self, input_tensor, frame_index, joint_index, person_index):
        # occlude single joint for a person over the temporal window centered at the frame
        B, C, T, V, M = input_tensor.shape

        if self.use_temporal_window:
            num_frames = T 
            start, end = self._get_window_bounds(frame_index, num_frames)
            occluded = input_tensor.clone()
            occluded[:, :, start:end, joint_index, person_index] = 0 # set one joint for one person around the temporal window to zero
        else:
            occluded = input_tensor.clone()
            occluded[:, :, frame_index, joint_index, person_index] = 0
        return occluded

    def _normalise_saliency_map(self, saliency):
        if self.normalisation_mode == "per_frame":
            saliency_min = saliency.flatten(1).min(dim=1).values.view(-1, 1, 1)
            saliency_max = saliency.flatten(1).max(dim=1).values.view(-1, 1, 1)
        elif self.normalisation_mode == "per_video":
            saliency_min = saliency.min()
            saliency_max = saliency.max()
        else:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")
        
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        return saliency

    def generate_heatmap(self, input_tensor, target_class=None):
        # input tensor [1, 3, 150, 17, 4]

        with torch.inference_mode():

            B, C, T, V, M = input_tensor.shape
            assert B == 1 # explain one video at a time

            baseline_logits, baseline_probabilities = self._predict(input_tensor)

            if target_class is None:
                target_class = baseline_probabilities.argmax(dim=1).item() # selects the models predicted class

            baseline_target_probability = baseline_probabilities[0, target_class] # unoccluded prediction confidence (reference point)

            joint_importance = torch.zeros(T, V, M, device=input_tensor.device)

            # only consider detected joints
            has_confidence = input_tensor[0, 2] > 0 # shape [T, V, M]

            # joint importance
            for frame_index in range(T):
                for person_index in range(M):
                    for joint_index in range(V):
                        if not has_confidence[frame_index, joint_index, person_index]: continue # no joint tetected

                        occluded_input = self._occlude_joint(input_tensor, frame_index, joint_index, person_index) # occlude the joint over temporal window
                        _, occluded_probability = self._predict(occluded_input) # calculate prediction confidence of occluded input
                        occluded_target_probability = occluded_probability[0, target_class]
                        joint_importance[frame_index, joint_index, person_index] = baseline_target_probability - occluded_target_probability # save prediction confidence 

            joint_importance = F.relu(joint_importance)
            joint_importance = self._normalise_saliency_map(joint_importance)

            return joint_importance.detach()