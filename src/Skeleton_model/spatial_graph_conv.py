import torch 
import torch.nn as nn

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions=3):
        '''
        in_channels: number of channels in the input data 
        out_channels: number of channels produced by the convolution
        num_partitions = 3 (root, centripetal, centrifugal)
        '''
        super().__init__()

        self.out_channels = out_channels
        self.num_partitions = num_partitions
        self.in_channels = in_channels

        # create separate feature maps for each partition (f_in * w_j)
        self.channel_transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * num_partitions, 
            kernel_size=1
        )

    def forward(self, x, adjacency):
        '''
            x shape [B, C, T, V]
                where:
                    B = Batch size
                    C = 3 (normalised_X_coordinate, normalised_Y_coordinate, confidence)
                    T = 150 (number of frames)
                    V = 17 (number of keypoints)

            adjacency shape [K, V, V]        
                where:
                    k = 3 (number of partitions)
                    V = 17 (number of keypoints)
            
            
            returns shape [B, C_out, T, V]
                where:
                    B = Batch size
                    C_out = out_channels
                    T = 150 (number of frames)
                    V = 17 (number of keypoints)
        '''

        x = self.channel_transform(x) # [B, K*C_out, T, V]
        B, _, T, V = x.shape

        x = x.reshape(
            B,
            self.num_partitions,
            self.out_channels, 
            T, 
            V
        )

        f_outs = []
        for k in range(self.num_partitions):
            output = torch.matmul(x[:, k], adjacency[k])
            f_outs.append(output)

        f_out = sum(f_outs)

        return f_out



        





        