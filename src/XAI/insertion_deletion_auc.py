import torch
import math
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt


class AUC_TEST:
    def __init__(self, model, data_type="image", score_mode="probability", num_steps=100, blur_kernel_size=31, blur_sigma=8.0):
        self.model = model
        self.score_mode = score_mode
        self.num_steps = num_steps

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.data_type = data_type

        if self.data_type not in ["image", "skeleton"]:
            raise ValueError("data_type must be either 'image' OR 'skeleton'")
        
    def _get_target_class(self, input_tensor, target_class):
        # input_tensor shape:
        # image: [1, 32, 3, 224, 224]
        # skeleton: [1, 3, 150, 17, 4]

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item() # target_class is predicted class
        
        return target_class, logits

    def _get_score(self, logits, target_class):
        if self.score_mode == "probability":
            probabilities = torch.softmax(logits, dim=1)
            return probabilities[0, target_class].item() # use probabilities in range 0-1
        else:
            return logits[0, target_class].item() # use raw logits 
    
    def _blur_video(self, video):
        # video: [1, 32, 3, 224, 224]

        B, T, C, H, W = video.shape
        frames = video.reshape(B*T, C, H, W)
        blurred_frames = gaussian_blur(
            frames, 
            kernel_size=[
                self.blur_kernel_size, 
                self.blur_kernel_size
            ],
            sigma=[
                self.blur_sigma,
                self.blur_sigma
            ]
        )
        blurred_video = blurred_frames.reshape(B, T, C, H, W)
        return blurred_video
    
    def _get_saliency_map_ranking(self, saliency_map):
        # saliency_map shape:
        # image: [32, 224, 224]
        # skeleton: [150, 17, 4]

        # sorted_indices[0] is the largest activation
        flat_map = saliency_map.flatten()
        sorted_indices = torch.argsort(flat_map, descending=True)
        total_cells = sorted_indices.numel() 
        cells_per_step = math.ceil(total_cells/self.num_steps)

        return sorted_indices, total_cells, cells_per_step
    
    def _flat_indices_to_saliency_map_coordinates(self, flat_indices, H, W):
        # image: H=224, W=224
        # skeleton: H=17, W=4

        cells_per_frame = H*W
        frame_indices = flat_indices // cells_per_frame # gets the frame number (0-31) OR (0-149)
        position_in_frame = flat_indices % cells_per_frame # gets which pixel inside the frame (0 - 223*223) OR (0 - 17*3)
        row_indices = position_in_frame // W # gets the row inside the frame (0-223) or (0-16)
        col_indices = position_in_frame % W # gets the col inside the frame (0-223) or (0-3)

        return frame_indices, row_indices, col_indices

    def _get_replacement_input_tensor(self, input_tensor, strategy, constant_value=0.0):

        if strategy == "blurred":
            return self._blur_video(input_tensor)
        elif strategy == "constant":
            return torch.full_like(input_tensor, fill_value=constant_value)
        else:
            raise ValueError("strategy must be either 'blurred' or 'constant'")
    
    def _calc_auc(self, scores, fractions):
        return torch.trapz(y=scores, x=fractions).item()

    def deletion(self, input_tensor, saliency_map, target_class=None):
        # iamge:
        # saliency_map shape [32, 224, 224]
        # input_tensor shape [1, 32, 3, 224, 224]

        # skeleton:
        # saliency_map shape [150, 17, 4]
        # input_tensor shape [1, 3, 150, 17, 4]

        target_class, original_logits = self._get_target_class(input_tensor, target_class)
        replacement = self._get_replacement_input_tensor(input_tensor, strategy="constant") # the replacement tensor
        sorted_indices, total_cells, cells_per_step = self._get_saliency_map_ranking(saliency_map)

        working_tensor = input_tensor.clone() # the starting tensor
        scores = [] # probability of target class at each step
        fractions = [] # percentage of removed pixels
        self.model.eval()

        if self.data_type == "image":
            B, T, C, H, W = input_tensor.shape
        else:
            B, C, T, V, M = input_tensor.shape

        with torch.inference_mode():
            initial_score = self._get_score(original_logits, target_class)
            scores.append(initial_score)
            fractions.append(0.0)

            l=0

            while l < total_cells:
                r = min(l+cells_per_step, total_cells)
                indices_to_replace = sorted_indices[l:r]

                if self.data_type == "image":
                    frame_idx, row_idx, col_idx = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=H, W=W)
                    working_tensor[0, frame_idx, :, row_idx, col_idx] = replacement[0, frame_idx, :, row_idx, col_idx]
                else:
                    frame_idx, joint_idx, person_idx = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=V, W=M)
                    working_tensor[0, :, frame_idx, joint_idx, person_idx] = replacement[0, :, frame_idx, joint_idx, person_idx]

                logits = self.model(working_tensor)
                score = self._get_score(logits, target_class)
                scores.append(score)
                fractions.append(r/total_cells)
                l=r
        
        scores = torch.tensor(scores, device=input_tensor.device)
        fractions = torch.tensor(fractions, device=input_tensor.device)
        auc = self._calc_auc(scores, fractions)

        return fractions, scores, auc
    
    def insertion(self, input_tensor, saliency_map, target_class=None):

        if self.data_type == "image":
            strategy = "blurred"
            B, T, C, H, W = input_tensor.shape
        else:
            strategy = "constant"
            B, C, T, V, M = input_tensor.shape

        target_class, original_logits = self._get_target_class(input_tensor, target_class)
        working_tensor = self._get_replacement_input_tensor(input_tensor, strategy=strategy)
        sorted_indices, total_cells, cells_per_step = self._get_saliency_map_ranking(saliency_map)

        scores = [] # probability of target class at each step
        fractions = [] # percentage of removed pixels
        self.model.eval()

        with torch.inference_mode():
            starting_tensor_logits = self.model(working_tensor)
            initial_score = self._get_score(starting_tensor_logits, target_class)
            scores.append(initial_score)
            fractions.append(0.0)

            l=0
            while l < total_cells:
                r = min(l+cells_per_step, total_cells)
                indices_to_replace = sorted_indices[l:r]

                if self.data_type == "image":
                    frame_idx, row_idx, col_idx = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=H, W=W)
                    working_tensor[0, frame_idx, :, row_idx, col_idx] = input_tensor[0, frame_idx, :, row_idx, col_idx]
                else:
                    frame_idx, joint_idx, person_idx = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=V, W=M)
                    working_tensor[0, :, frame_idx, joint_idx, person_idx] = input_tensor[0, :, frame_idx, joint_idx, person_idx]

                logits = self.model(working_tensor)
                score = self._get_score(logits, target_class)
                scores.append(score)
                fractions.append(r/total_cells)
                l=r
        
        scores = torch.tensor(scores, device=input_tensor.device)
        fractions = torch.tensor(fractions, device=input_tensor.device)
        auc = self._calc_auc(scores, fractions)

        return fractions, scores, auc

    @staticmethod
    def plot_auc_curve(fractions, scores, auc, metric, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
 
        fractions = fractions.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
 
        if label is None:
            label = f"{metric} (AUC = {auc:.3f})"
 
        ax.plot(fractions, scores, linewidth=2, label=label)
        ax.fill_between(fractions, scores, alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of locations perturbed")
        ax.set_ylabel("Target class probability")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
 
        return ax