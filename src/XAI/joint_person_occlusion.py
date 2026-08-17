import torch
import torch.nn.functional as F

class JointOcclusion:
    def __init__(self, model, temporal_window_size=9):
        self.model = model 
        self.temporal_window_size = temporal_window_size
        self.model.eval()

        assert self.temporal_window_size % 2 != 0

    @torch.inference_mode()
    def _predict(self, input_tensor):
        '''
        input_tensor: [1, C, T, V, M]
        returns:
            logits [1, 2]
            probabilities [1, 2]
        '''

        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities

    def _get_window_bounds(self, frame_index, num_frames):
        half_window = self.temporal_window_size // 2
        start = max(0, frame_index - half_window)
        end = min(num_frames, frame_index + half_window + 1)

        return start, end
    
    def _occlude_joint(self, input_tensor, frame_index, joint_index, person_index):
        '''
        returns a copy of the input with joint removed from one person inside temporal window centered around frame_index
        '''
        B, C, T, V, M = input_tensor.shape
        num_frames = T 
        start, end = self._get_window_bounds(frame_index, num_frames)
        occluded = input_tensor.clone()
        occluded[:, :, start:end, joint_index, person_index] = 0 # set one joint for one person around the temporal window to zero
        return occluded

    def _occlude_person(self, input_tensor, frame_index, person_index):
        '''
        returns a copy of the input with one person removed completely from temporal window centered around frame_index
        '''
        B, C, T, V, M = input_tensor.shape
        num_frames = T 
        start, end = self._get_window_bounds(frame_index, num_frames)
        occluded = input_tensor.clone()
        occluded[:, :, start:end, :, person_index] = 0
        return occluded


    @torch.inference_mode()
    def explain(self, input_tensor, target_class=None):
        '''
            calculate joint and person importance (baseline importance - occluded importance)
            returns:
                joint_importance: shape [T, V, M]
                person_importance: shape[T, M]
        '''
        B, C, T, V, M = input_tensor.shape
        assert B == 1 # explain one video at a time

        baseline_logits, baseline_probabilities = self._predict(input_tensor)

        if target_class is None:
            target_class = baseline_probabilities.argmax(dim=1).item() # selects the models predicted class

        baseline_target_probability = baseline_probabilities[0, target_class] # unperturbed prediction confidence (reference point)

        joint_importance = torch.zeros(T, V, M, device=input_tensor.device)
        person_importance = torch.zeros(T, M, device=input_tensor.device)

        # joint importance
        for frame_index in range(T):
            for person_index in range(M):
                for joint_index in range(V):
                    occluded_input = self._occlude_joint(input_tensor, frame_index, joint_index, person_index) # occlude the joint over temporal window
                    _, occluded_probability = self._predict(occluded_input) # calculate prediction confidence of occluded input
                    occluded_target_probability = occluded_probability[0, target_class]
                    joint_importance[frame_index, joint_index, person_index] = baseline_target_probability - occluded_target_probability # save prediction confidence 

        # person importance
        for frame_index in range(T):
            for person_index in range(M):
                occluded_input = self._occlude_person(input_tensor, frame_index, person_index)
                _, occluded_probability = self._predict(occluded_input)
                occluded_target_probability = occluded_probability[0, target_class]
                person_importance[frame_index, person_index] = baseline_target_probability - occluded_target_probability

        return {
            "baseline_logits": baseline_logits,
            "baseline_probabilities": baseline_probabilities,
            "predicted_class": baseline_probabilities.argmax(dim=1).item(),
            "target_class": target_class,
            "joint_importance": joint_importance, # [T, V, M]
            "person_importance": person_importance, # [T, M]
        }