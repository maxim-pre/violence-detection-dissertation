from collections import defaultdict
import torch

def pose_data_to_stgcn_tensor(pose_data, num_frames=150, num_keypoints=17, max_people=2):
    '''
    convert pose data -> (C, T, V, M)

    where: 
        C = (normalised_X_coordinate, normalised_Y_coordinate, confidence)
        T = number of frames
        V = number of keypoints
        M = maximum number of tracked people
    '''

    H, W = pose_data["original_video_shape"]

    # 1. rank each track so we can choose the top two tracks
    # will rank the tracks by duration * confidence. i.e. (number of frames appeard * average confidence of detected joints)

    track_frame_count = defaultdict(int)
    track_keypoint_count = defaultdict(int)
    track_keypoint_confidence_sum = defaultdict(float)

    for frame in pose_data["frames"]:
        for person in frame["people"]:

            track_id = int(person["track_id"])
            confidences = person["keypoint_confidence"].float()

            track_frame_count[track_id] += 1
            track_keypoint_count[track_id] += num_keypoints
            track_keypoint_confidence_sum[track_id] += confidences.sum().item()

    # compute track score
    track_scores = {}
    for track_id in track_frame_count:
        mean_confidence = track_keypoint_confidence_sum[track_id] / track_keypoint_count[track_id]
        track_scores[track_id] = track_frame_count[track_id] * mean_confidence

    # select the strongest tracks according to max_people
    sorted_tracks = sorted(track_scores.items(), key=lambda item: item[1], reverse=True)
    selected_track_ids = [track_id for track_id, score in sorted_tracks[:max_people]]

    track_to_person_index = {}
    for person_index, track_id in enumerate(selected_track_ids):
        track_to_person_index[track_id] = person_index

    # construct the tensor
    tensor = torch.zeros(3, num_frames, num_keypoints, max_people, dtype=torch.float32)

    for frame in pose_data["frames"]:
        frame_index = frame["frame_index"]
        if not 0 <= frame_index < num_frames: continue

        for person in frame["people"]:
            
            track_id = person["track_id"]
            if track_id not in track_to_person_index: continue 

            person_index = track_to_person_index[track_id]
            keypoints = person["keypoints"]
            confidences = person["keypoint_confidence"]

            # normalise between 0-1
            x = keypoints[:, 0] / W
            y = keypoints[:, 1] / H

            tensor[0, frame_index, :, person_index] = x
            tensor[1, frame_index, :, person_index] = y
            tensor[2, frame_index, :, person_index] = confidences

    return tensor


