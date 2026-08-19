
import torch

'''
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
Sijie Yan, Yuanjun Xiong, Dahua Lin

@inproceedings{stgcn2018aaai,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}

'''

def compute_joint_distance_to_center_of_gravity(dataset):

    distances_sum = torch.zeros(17)
    count = torch.zeros(17)
    for data, label in dataset:
        C, T, V, M = data.shape

        xy = data[:2]
        conf = data[2]

        num_frames = T
        num_people = M

        for frame_index in range(num_frames):
            for person in range(num_people):

                # 
                valid_keypoints = conf[frame_index, :, person] > 0.1 # only consider keypoints with confidence score > 0.1
                if valid_keypoints.sum() < 12: # don't contribute skeletons to the average if there are less that 12 somewhat confident keypoints
                    continue

                joints = xy[:, frame_index, valid_keypoints, person].T 
                centre = joints.mean(dim=0)
                distances = torch.norm(joints - centre, dim=1)

                distances_sum[valid_keypoints] += distances
                count[valid_keypoints] += 1

    return distances_sum/count



class SkeletonGraph:
    def __init__(self, radii, normalisation="column"):
        '''
        radii: at index i = average distance of joint i from the center of gravity
        '''
        self.radii = torch.tensor(radii, dtype=torch.float32)
        self.num_joints = 17
        self.normalisation = normalisation
        
        # Skeleton edges taken directly from ultralytics
        self.skeleton = [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ]

        self.A = self._build_adjacency_matrix_with_spatial_partitioning()

    def _symmetric_normalise_adjacency_matrix(self, full_adjacency):
        # full_adjacency shape [17, 17]
        # A_hat = D^(-1/2) @ A @ D^(-1/2)
        # paper states symmetric normalisation while released implementation uses column normalisation

        degree = full_adjacency.sum(dim=1)
        inverse_sqrt_degree = torch.zeros_like(degree)
        non_zero = degree > 0
        inverse_sqrt_degree[non_zero] = degree[non_zero].pow(-0.5)
        degree_matrix = torch.diag(inverse_sqrt_degree)

        return degree_matrix @ full_adjacency @ degree_matrix

    def _column_normalise_adjacency_matrix(self, full_adjacency):
        # full_adjacency shape [17, 17]

        degree = full_adjacency.sum(dim=0) # sum the columns (how many connections each joint has)
        inverse_degree = torch.zeros_like(degree)
        non_zero = degree > 0 
        inverse_degree[non_zero] = 1 / degree[non_zero] # calculate the reciprocal
        degree_matrix = torch.diag(inverse_degree) # convert to diagonal matrix D^-1
        return full_adjacency @ degree_matrix # A*D^-1

    def _normalise_adjacency_matrix(self, adjacency_matrix):
        if self.normalisation == "column":
            return self._column_normalise_adjacency_matrix(adjacency_matrix)
        elif self.normalisation == "symmetric":
            return self._symmetric_normalise_adjacency_matrix(adjacency_matrix)
        else:
            raise ValueError("normalisation must be column or symmetric")

    def _build_adjacency_matrix_with_spatial_partitioning(self):
        # return adjacency matrix: [3, 17, 17]
        # root: self-connections -> rj = ri
        # centipetal: neighbouring joint is closer to the center of gravity -> rj < ri
        # centrifugal: neighbouting joint is farther from the center of gravity -> rj > ri

        root = torch.zeros(self.num_joints, self.num_joints)
        centripetal = torch.zeros(self.num_joints, self.num_joints)
        centrifugal = torch.zeros(self.num_joints, self.num_joints)

        root.fill_diagonal_(1) # every joint is connected to itself (self-connections)

        for joint_i, joint_j in self.skeleton:
            ri = self.radii[joint_i]
            rj = self.radii[joint_j]

            if rj < ri:
                centripetal[joint_i, joint_j] = 1
                centrifugal[joint_j, joint_i] = 1

            elif rj > ri:
                centrifugal[joint_i, joint_j] = 1
                centripetal[joint_j, joint_i] = 1
            else:
                root[joint_i, joint_j] = 1
                root[joint_j, joint_i] = 1


        full_adjacency = root + centripetal + centrifugal
        normalised_full_adjacency = self._normalise_adjacency_matrix(full_adjacency)

        normalised_root = root * normalised_full_adjacency
        normalised_centripetal = centripetal * normalised_full_adjacency
        normalised_centrifugal = centrifugal * normalised_full_adjacency

        # for debugging and stuff
        self.root = root 
        self.centripetal = centripetal
        self.centrifugal = centrifugal
        self.normalised_root = normalised_root
        self.normalised_centripetal = normalised_centripetal
        self.normalised_centrifugal = normalised_centrifugal
        self.normalised_full_adjacency = normalised_full_adjacency
        self.full_adjacency = full_adjacency


        return torch.stack([normalised_root, normalised_centripetal, normalised_centrifugal], dim=0)
        





