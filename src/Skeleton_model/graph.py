
import torch

'''
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
Sijie Yan, Yuanjun Xiong, Dahua Lin

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
    def __init__(self, radii):
        '''
        radii: at index i = average distance of joint i from the center of gravity
        '''
        self.radii = torch.tensor(radii, dtype=torch.float32)
        self.num_joints = 17
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

    def _normalise_adjacency_matrix(self, adjacency_matrix, alpha=0.001):
        '''
        A_hat = D_j^(-1/2) @ A_j @ D_j^(-1/2)
        D_j[i, i] = sum_k A_j[i, k] + alpha
            where:
                A_j is one partition of the graph (root, centripetal, centrifugal)
                D is the degree matrix constructed from A
        '''

        degree = adjacency_matrix.sum(dim=1) + alpha # counts connections of joints (len 17)
        inverse_sqrt_degree = degree.pow(-0.5)
        degree_matrix = torch.diag(inverse_sqrt_degree)

        return degree_matrix @ adjacency_matrix @ degree_matrix

    def _build_adjacency_matrix_with_spatial_partitioning(self):
        '''
        return adjacency matrix: [3, 17, 17]

        root: self-connections -> rj = ri
        centipetal: neighbouring joint is closer to the center of gravity -> rj < ri
        centrifugal: neighbouting joint is farther from the center of gravity -> rj > ri
        '''
        root = torch.zeros(self.num_joints, self.num_joints)
        centripetal = torch.zeros(self.num_joints, self.num_joints)
        centrifugal = torch.zeros(self.num_joints, self.num_joints)

        root.fill_diagonal_(1) # every joint is connected to itself

        for joint_i, joint_j in self.skeleton:
            ri = self.radii[joint_i]
            rj = self.radii[joint_j]

            if rj < ri:
                centripetal[joint_i, joint_j] = 1
                centrifugal[joint_j, joint_i] = 1

            elif rj > ri:
                centrifugal[joint_i, joint_j] = 1
                centripetal[joint_j, joint_i] = 1

        self.unormalised_root = root
        self.unormalised_centripetal = centripetal
        self.unormalised_centrifugal = centrifugal

        root = self._normalise_adjacency_matrix(root)
        centripetal = self._normalise_adjacency_matrix(centripetal)
        centrifugal = self._normalise_adjacency_matrix(centrifugal)

        return torch.stack([root, centripetal, centrifugal], dim=0)




