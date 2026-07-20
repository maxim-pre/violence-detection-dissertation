import torch
import math
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt


class AUC_TEST:
    def __init__(self, model, score_mode="probability", num_steps=100, blur_kernel_size=31, blur_sigma=8.0):
        self.model = model
        self.score_mode = score_mode
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.num_steps = num_steps
        
    def _get_target_class(self, video, target_class):
        '''
        video: [1, 32, 3, 224, 224]
        '''

        self.model.eval()

        with torch.no_grad():
            logits = self.model(video)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item() # target_class is predicted class
        
        return target_class, logits

    def _get_score(self, logits, target_class):
        if self.score_mode == "probability":
            probabilities = torch.softmax(logits, dim=1)
            return probabilities[0, target_class] # use probabilities in range 0-1
        else:
            return logits[0, target_class] # use raw logits 
    
    def _blur_video(self, video):

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
        '''
        saliency_map [32, 224, 224]
        '''
        # sorted_indices[0] is the largest activation
        flat_map = saliency_map.flatten()
        sorted_indices = torch.argsort(flat_map, descending=True)
        total_pixels = sorted_indices.numel() 
        pixels_per_step = math.ceil(total_pixels/self.num_steps)

        return sorted_indices, total_pixels, pixels_per_step
    
    def _flat_indices_to_saliency_map_coordinates(self, flat_indices, H, W):
        pixels_per_frame = H*W
        frame_indices = flat_indices // pixels_per_frame # gets the frame number (0-31)
        frame_pixels = flat_indices % pixels_per_frame # gets which pixel it is in side the frame (0 - 223*223)
        row_indices = frame_pixels // W # gets the row inside the frame (0-223)
        col_indices = frame_pixels % W # gets the col inside the frame (0-223)

        return frame_indices, row_indices, col_indices

    def _get_replacement_video(self, video, strategy, constant_value=0.0):

        if strategy == "blurred":
            return self._blur_video(video)
        elif strategy == "constant":
            return torch.full_like(video, fill_value=constant_value)
        else:
            raise ValueError("strategy must be either 'blurred' or 'constant'")
    
    def _calc_auc(self, scores, fractions):
        return torch.trapz(y=scores, x=fractions).item()

    def deletion(self, video, saliency_map, target_class=None, constant_value=0.0, strategy="constant"):
        '''
        saliency_map [32, 224, 224]
        video [1, 32, 3, 224, 224]
        '''

        target_class, original_logits = self._get_target_class(video, target_class)
        replacement = self._get_replacement_video(video, strategy=strategy, constant_value=constant_value) # the replacement image
        B, T, C, H, W = video.shape

        sorted_indices, total_pixels, pixels_per_step = self._get_saliency_map_ranking(saliency_map)

        working_video = video.clone()
        scores = [] # probability of target class at each step
        fractions = [] # percentage of removed pixels
        self.model.eval()

        with torch.no_grad():
            initial_score = self._get_score(original_logits, target_class)
            scores.append(initial_score)
            fractions.append(0.0)

            l=0

            while l < total_pixels:
                r = min(l+pixels_per_step, total_pixels)
                indices_to_replace = sorted_indices[l:r]
                frame_indices, row_indices, col_indices = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=H, W=W)
                working_video[0, frame_indices, :, row_indices, col_indices] = replacement[0, frame_indices, :, row_indices, col_indices]

                logits = self.model(working_video)
                score = self._get_score(logits, target_class)
                scores.append(score)
                fractions.append(r/total_pixels)
                l=r
        
        scores = torch.stack(scores)
        fractions = torch.tensor(fractions, device=video.device)
        auc = self._calc_auc(scores, fractions)

        return {
            "metric": "Deletion", 
            "auc": auc, 
            "scores": scores, 
            "fractions": fractions, 
            "original_logits": original_logits,
            "target_class": target_class, 
            "pixels_per_step": pixels_per_step,
            "total_pixels": total_pixels, 
            "strategy": strategy, 
        }
    
    def insertion(self, video, saliency_map, target_class=None, strategy="blurred"):
        target_class, original_logits = self._get_target_class(video, target_class)
        starting_video = self._get_replacement_video(video, strategy=strategy)
        B, T, C, H, W = video.shape

        sorted_indices, total_pixels, pixels_per_step = self._get_saliency_map_ranking(saliency_map)
        working_video = starting_video.clone()

        scores = [] # probability of target class at each step
        fractions = [] # percentage of removed pixels
        self.model.eval()

        with torch.no_grad():
            starting_video_logits = self.model(starting_video)
            initial_score = self._get_score(starting_video_logits, target_class)
            scores.append(initial_score)
            fractions.append(0.0)

            l=0
            while l < total_pixels:
                r = min(l+pixels_per_step, total_pixels)
                indices_to_replace = sorted_indices[l:r]
                frame_indices, row_indices, col_indices = self._flat_indices_to_saliency_map_coordinates(indices_to_replace, H=H, W=W)
                working_video[0, frame_indices, :, row_indices, col_indices] = video[0, frame_indices, :, row_indices, col_indices]

                logits = self.model(working_video)
                score = self._get_score(logits, target_class)
                scores.append(score)
                fractions.append(r/total_pixels)
                l=r
        
        scores = torch.stack(scores)
        fractions = torch.tensor(fractions, device=video.device)
        auc = self._calc_auc(scores, fractions)

        return {
            "metric": "Insertion", 
            "auc": auc, 
            "scores": scores, 
            "fractions": fractions, 
            "original_logits": original_logits,
            "target_class": target_class, 
            "pixels_per_step": pixels_per_step,
            "total_pixels": total_pixels, 
            "strategy": strategy, 
        }

    @staticmethod
    def plot_auc_curve(results, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        fractions = results["fractions"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()

        if label is None:
            label = f"{results["metric"]} (AUC = {results['auc']:.3f})"

        ax.plot(fractions, scores, linewidth=2, label=label)
        ax.fill_between(fractions,scores,alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of locations perturbed")
        ax.set_ylabel(f"Target class probability")
        ax.grid(True, linestyle="--", alpha=0.5)

        return ax


    











    

    

    

    




        
        
    

    


        
