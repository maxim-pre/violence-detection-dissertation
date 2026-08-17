import torch
import torch.nn.functional as F
import math

class RISE:
    def __init__(self, model, num_masks=1000, mask_batch_size=8, temporal_grid_size=8, spatial_grid_size=7, normalisation_mode="per_frame"):
        self.model = model
        self.num_masks = num_masks
        self.mask_batch_size = mask_batch_size
        self.temporal_grid_size = temporal_grid_size
        self.spatial_grid_size = spatial_grid_size
        self.mask_probability = 0.5
        self.normalisation_mode = normalisation_mode

        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")


    def _generate_masks(self, number_of_masks, num_frames, height, width, device):
        # 1. generate coarse masks 8x7x7
        # 2. upscale to larger than input image 36x256x256
        # 3. crop to original size 32x224x224

        temporal_cell_size = num_frames // self.temporal_grid_size # 32 // 8 = 4
        height_cell_size = height // self.spatial_grid_size # 224 // 7 = 32
        width_cell_size = width // self.spatial_grid_size

        # shape [num_masks, 8, 7, 7]
        coarse_masks = (
            torch.rand(
                number_of_masks, 
                self.temporal_grid_size,
                self.spatial_grid_size, 
                self.spatial_grid_size,
                device=device
            ) < self.mask_probability
        ).float()

        # shape [num_masks, 1, 8, 7, 7]
        coarse_masks = coarse_masks.unsqueeze(1)

        enlarged_temporal_size = (1 + self.temporal_grid_size) * temporal_cell_size # (1 + 8) * 4 = 36
        enlarged_height = (1 + self.spatial_grid_size) * height_cell_size # (1 + 7) * 32 = 256
        enlarged_width = (1 + self.spatial_grid_size) * width_cell_size

        # shape [num_masks, 1, 36, 256, 256]
        enlarged_masks = F.interpolate(
            coarse_masks, 
            size=(
                enlarged_temporal_size, 
                enlarged_height, 
                enlarged_width
            ),
            mode="trilinear",
            align_corners=False,
        ) 

        # shape [num_masks, 36, 256, 256]
        enlarged_masks = enlarged_masks.squeeze(1)

        masks = torch.empty(
            number_of_masks, 
            num_frames,
            height, 
            width, 
            device=device
        )

        # now cropping to shape [32, 224, 244]
        for i in range(number_of_masks):
            temporal_offset = torch.randint(low=0, high=temporal_cell_size, size=(1,), device=device).item()
            height_offset = torch.randint(low=0, high=height_cell_size, size=(1,), device=device).item()
            width_offset = torch.randint(low=0, high=width_cell_size, size=(1,), device=device).item()

            masks[i] = enlarged_masks[
                i,
                temporal_offset: num_frames + temporal_offset,
                height_offset: height + height_offset, 
                width_offset: width + width_offset
            ]
        
        return masks

    def _normalise_saliency_map(self, saliency):
        '''
            saliency shape: [32, 7, 7] or [batch_size * num_frames, 7, 7] 
        '''
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

        B, T, C, H, W = input_tensor.shape # 1, 32, 3, 224, 224
        self.model.eval()

        with torch.inference_mode():
            logits = self.model(input_tensor)
        
        if target_class == None:
            target_class = logits.argmax(dim=1).item()

        saliency_accumulator = torch.zeros(T, H, W, device=input_tensor.device)
        masks_processed = 0 

        while masks_processed < self.num_masks:
            current_batch_size = min(self.mask_batch_size, self.num_masks - masks_processed)

            # shape [current_batch_size, 32, 224, 224]
            masks = self._generate_masks(
                number_of_masks=current_batch_size,
                num_frames=T, 
                height=H, 
                width=W,
                device=input_tensor.device
            )

            # [1,32,3,224,224] * [batch_size,32,1,224,224] = [batch_size,32,3,224,224]
            masked_videos = input_tensor * masks.unsqueeze(2) 

            with torch.inference_mode():
                masked_logits = self.model(masked_videos)
                probabilities = torch.softmax(masked_logits, dim=1)
                target_scores = probabilities[:, target_class]

            weighted_masks = target_scores[:, None, None, None] * masks
            saliency_accumulator += weighted_masks.sum(dim=0)
            masks_processed += current_batch_size
            print(f"processed {masks_processed}/{self.num_masks} masks")
        
        saliency_map = saliency_accumulator/(self.mask_probability * self.num_masks)
        print(f"saliency_map shape: {saliency_map.shape}")
        normalised_saliency_map = self._normalise_saliency_map(saliency_map)

        return (
            saliency_map.detach(),
            normalised_saliency_map.detach(),
            logits.detach(),
            target_class
        )

class SkeletonRise:
    def __init__(self, model, num_masks=1000, mask_batch_size=8, temporal_grid_size=17, normalisation_mode="per_frame"):

        self.model = model
        self.num_masks = num_masks
        self.mask_batch_size = mask_batch_size
        self.temporal_grid_size = temporal_grid_size
        self.mask_probability = 0.5 
        self.normalisation_mode = normalisation_mode

        if self.normalisation_mode not in ["per_frame", "per_video"]:
            raise ValueError("normalisation_mode must be either 'per_frame' OR 'per_video'")

    def _generate_temporal_masks(self, number_of_masks, num_frames, device):

        temporal_cell_size = math.ceil(num_frames / self.temporal_grid_size) # 150 / 17 = 9

        # shape: [num_masks, 17]
        coarse_masks = (
            torch.rand(
                number_of_masks, 
                self.temporal_grid_size, 
                device=device
            ) < self.mask_probability
        ).float()

        # shape: [num_masks, 1, 17]
        coarse_masks = coarse_masks.unsqueeze(1)

        enlarged_temporal_size = (1 + self.temporal_grid_size) * temporal_cell_size # 18 * 9 = 162
        enlarged_masks = F.interpolate(coarse_masks, size=enlarged_temporal_size, mode="linear", align_corners=False) 
        enlarged_masks = enlarged_masks.squeeze(1) # shape [num_masks, 162]

        temporal_masks = torch.empty(number_of_masks, num_frames, device=device) # shape [num_masks, 150]

        # cropping to shape [num_masks, 150]
        for mask in range(number_of_masks):
            offset = torch.randint(low=0, high=temporal_cell_size, size=(1,), device=device).item()
            temporal_masks[mask] = enlarged_masks[mask, offset: num_frames + offset]

        return (temporal_masks > 0.5).float() # need to convert back to binary as we're dealing with coordinates not pixels

    def _generate_masks(self, number_of_masks, num_frames, num_joints, num_people, present_people, device):
        # return masks of shape [num_masks, T, V, M]

        masks = torch.zeros(number_of_masks, num_frames, num_joints, num_people, device=device)

        # only need masks for non-empty tracks
        for person in present_people:
            temporal_mask = self._generate_temporal_masks(number_of_masks, num_frames, device) # which frames to apply joint mask

            # shape [num_masks, 17]
            joint_mask = (
                torch.rand(number_of_masks, num_joints, device=device) < self.mask_probability
            ).float()

            joint_mask = joint_mask.unsqueeze(1) # shape [num_masks, 1, 17]
            temporal_mask = temporal_mask.unsqueeze(2) # shape [num_masks, 150, 1]

            masks[:, :, :, person] = temporal_mask * joint_mask # shape [num_masks, 150, 17]


        return masks # shape [num_masks, 150, 17, 4]

    def _normalise_saliency_map(self, saliency):
        '''
            saliency shape: [150, 17, 4]
        '''
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
        # input tensor shape : [1, 3, 150, 17, 4]

        B, C, T, V, M = input_tensor.shape
        self.model.eval()

        with torch.inference_mode():
            logits = self.model(input_tensor)

        if target_class == None:
            target_class = logits.argmax(dim=1).item()


        # calculate how many people are actually present 
        has_confidence = input_tensor[0, 2] > 0 # shape [T, V, M]
        present_in_frame = has_confidence.any(dim=1) # shape [T, M]
        presence = present_in_frame.any(dim=0) # shape [M]
        present_people = [person for person in range(M) if presence[person]]

        saliency_accumulator = torch.zeros(T, V, M, device=input_tensor.device)
        masks_processed = 0 

        while masks_processed < self.num_masks:
            current_batch_size = min(self.mask_batch_size, self.num_masks - masks_processed)

            # shape [current_batch_size, 150, 17, num_present]
            masks = self._generate_masks(
                number_of_masks=current_batch_size,
                num_frames=T, 
                num_joints=V, 
                num_people=M,
                present_people=present_people, 
                device=input_tensor.device
            )

            # [1, 3, 150, 17, 4] * [curr_batch_size, 1, 150, 17, 4] = [curr_batch_size, 3, 150, 17, 4]
            masked_tensor = input_tensor * masks.unsqueeze(1)

            with torch.inference_mode():
                masked_logits = self.model(masked_tensor)
                probabilites = torch.softmax(masked_logits, dim=1)
                target_scores = probabilites[:, target_class] 

            weighted_masks = target_scores[:, None, None, None] * masks
            saliency_accumulator += weighted_masks.sum(dim=0)
            masks_processed += current_batch_size
            print(f"processed {masks_processed}/{self.num_masks} masks")


        saliency_map = saliency_accumulator/(self.mask_probability * self.num_masks)
        print(f"saliency_map shape: {saliency_map.shape}")
        normalised_saliency_map = self._normalise_saliency_map(saliency_map) # [150, 17, 4]
        return (
            saliency_map.detach(),
            normalised_saliency_map.detach(),
            logits.detach(),
            target_class
        )




        
    

    

        




        
                
            

            












        






