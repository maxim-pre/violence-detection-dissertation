import torch

class RIS_TEST:
    def __init__(self, model, data_type, noise_std=0.01, noise_mode="per_frame", epsilon_min=1e-6, attempts=50, ):
        self.model = model 
        self.data_type = data_type
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.epsilon_min = epsilon_min

        if self.noise_mode not in ["per_frame", "per_video"]:
            raise ValueError("noise_mode must be either 'per_frame' OR 'per_video'")

        if self.data_type not in ["image", "skeleton"]:
            raise ValueError("data_type must be either 'image' OR 'skeleton'")

    def _get_predicted_class(self, input_tensor):
        # input_tensor shape:
        # image: [1, 32, 3, 224, 224]
        # skeleton: [1, 3, 150, 17, 4]

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(input_tensor)
        
        predicted_class = logits.argmax(dim=1).item() # target_class is predicted class
        return predicted_class

    def _get_noisy_video(self, input_tensor):
        # image: [1, 32, 3, 224, 224]
        B, T, C, H, W = input_tensor.shape

        if self.noise_mode == "per_frame":
            noise = torch.randn(B, T, C, H, W, device=input_tensor.device) * self.noise_std
        
        elif self.noise_mode == "per_video":
            noise = torch.randn(B, 1, C, H, W, device=input_tensor.device) * self.noise_std
        else:
            raise ValueError("noise_mode must be either 'per_frame' OR 'per_video'")

        return input_tensor + noise

    def _get_noisy_skeleton(self, input_tensor):
        # skeleton: [1, 3, 150, 17, 4]
        B, C, T, V, M = input_tensor.shape

        has_confidence = (input_tensor[0, 2] > 0).float() # [T, V, M] 0/1 depending if joint present

        if self.noise_mode == "per_frame":
            noise = torch.randn(B, C, T, V, M, device=input_tensor.device) * self.noise_std
        elif self.noise_mode == "per_video":
            noise = torch.randn(B, C, 1, V, M, device=input_tensor.device) * self.noise_std
        else:
            raise ValueError("noise_mode must be either 'per_frame' OR 'per_video'")

        noise[:, 2] = 0 # don't add noise to the confidence channel 
        noise = noise * has_confidence # don't add noise to missing joints

        return input_tensor + noise

    def _get_noisy_input(self, input_tensor):
        if self.data_type == "image":
            return self._get_noisy_video(input_tensor)
        else:
            return self._get_noisy_skeleton(input_tensor)

    def _calc_relative_change(self, a, b):
        # returns single number 
        ratio = (a - b) / (a + self.epsilon_min) # deviation from (Agarwal, 2022) - need to add epsilon_min to denominator because skeleton_data contans zeros
        return torch.norm(ratio.flatten(), p=2)

    def evaluate(self, input_tensor, saliency_map, explain_fn, num_perturbations):

        target_class = self._get_predicted_class(input_tensor)
        ratios = []

        while len(ratios) < num_perturbations:
            noisy_input = self._get_noisy_input(input_tensor)
            predicted_class = self._get_predicted_class(noisy_input)

            if predicted_class != target_class: # skip if adding noise to the input changes its actual prediction
                continue

            noisy_saliency_map = explain_fn(noisy_input)
            saliency_change = self._calc_relative_change(saliency_map, noisy_saliency_map)
            input_change = self._calc_relative_change(input_tensor, noisy_input)
            input_change = torch.clamp(input_change, min=self.epsilon_min) # prevent divide by zero error 

            ratios.append((saliency_change/input_change).item())

        return max(ratios)




    



    