import torch 
import torch.nn as nn
import torch.nn.functional as F

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

        # equation 10 in paper
        f_outs = []
        for k in range(self.num_partitions):
            output = torch.matmul(x[:, k], adjacency[k])
            f_outs.append(output)

        f_out = sum(f_outs)

        return f_out

class STGCNBlock(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels, 
                num_partitions=3,
                temporal_kernel_size=9, 
                stride=1, 
                dropout=0, 
                residual=True,
            ):
        super().__init__()

        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd") # so there are an even number of frames before and after
        
        temporal_padding = (temporal_kernel_size - 1) // 2

        self.spatial_graph_conv = SpatialGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_partitions=num_partitions
        )

        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1), 
                padding=(temporal_padding, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        self.residual = self._build_residual_connection(
            in_channels=in_channels, 
            out_channels=out_channels, 
            stride=stride, 
            residual=residual
        )

        self.relu = nn.ReLU(inplace=True)

    def _build_residual_connection(self, in_channels, out_channels, stride, residual):
        if not residual:
            return lambda x: 0
        elif (in_channels == out_channels) and (stride == 1): # tensor shape is the same so input can be added directly
            return lambda x: x 
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=1, 
                    stride=(stride, 1), 
                    bias=False
                ), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, adjacency):
        residual_output = self.residual(x)
        x = self.spatial_graph_conv(x, adjacency)
        x = self.temporal_conv(x)
        x = x + residual_output
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self,
                 adjacency,
                 in_channels=3,
                 num_joints=17,
                 temporal_kernel_size=9,
                 dropout=0.5, 
                 edge_importance_weighting=True,
                 people_aggregation="masked_mean",
            ):
        '''
        adjacency shape: [K, V, V] wher K
        '''
        super().__init__()

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.data_batch_norm = nn.BatchNorm1d(in_channels * num_joints) # normalise each joint-channel feature across the batch and time (3*17 = 51 independent normalisations)
        self.people_aggregation = people_aggregation

        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, temporal_kernel_size=temporal_kernel_size, residual=False), 
            STGCNBlock(64, 64, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(64, 64, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(64, 64, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(64, 128, stride=2, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(128, 128, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(128, 128, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(128, 256, stride=2, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(256, 256, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(256, 256, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
        ])

        if edge_importance_weighting: # learn weights for each connection in the skeleton graph
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(self.adjacency)) for _ in self.stgcn_blocks
            ])
        else:
            self.edge_importance = [1.0 for _ in self.stgcn_blocks]

        self.classifier = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

    def forward(self, x):
        '''
        x shape: [B, C, T, V, M]
        where:
            B = Batch size
            C = input channels -> (X, Y, Confidence)
            T = Frames -> 150
            V = Joints -> 17
            M = people
        '''

        B, C, T, V, M = x.shape

        # 0 if person M in batch B has no detection anywhere in the clip
        presence = (x[:, 2] > 0).any(dim=1).any(dim=1).float()  # (B, M)

        x = x.permute(0, 4, 3, 2, 1).contiguous()
        x = x.view(B*M, V*C, T)
        x = self.data_batch_norm(x)
        x = x.view(B, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B*M, C, T, V)

        for block, importance in zip(self.stgcn_blocks, self.edge_importance):
            weighted_adjacency = self.adjacency * importance 
            x = block(x, weighted_adjacency)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(B, M, 256)

        if self.people_aggregation == "masked_mean":
            mask = presence.unsqueeze(-1) # (B, M, 1)
            summed = (x * mask).sum(dim=1) # (B, 256)
            count = mask.sum(dim=1).clamp(min=1.0) # (B, 1)
            x = summed / count

        elif self.people_aggregation == "max":
            # push absent people to -inf so they never win the max
            mask = presence.unsqueeze(-1).bool()                 # [B, M, 1]
            x = x.masked_fill(~mask, float("-inf"))
            x = x.max(dim=1).values                              # [B, 256]
            # max over all -inf gives -inf — clamp back to 0
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        else:
            raise ValueError("people_aggregation must be 'masked_mean' or 'max'")


        x = x.view(B, x.size(1), 1, 1)
        x = self.classifier(x)
        x = x.view(B, -1)

        return x


class STGCNV2(nn.Module):
    def __init__(self,
                 adjacency,
                 in_channels=3,
                 num_joints=17,
                 temporal_kernel_size=9,
                 dropout=0.5, 
                 edge_importance_weighting=True,
            ):
        '''
        adjacency shape: [K, V, V] wher K
        '''
        super().__init__()

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.data_batch_norm = nn.BatchNorm1d(in_channels * num_joints) # normalise each joint-channel feature across the batch and time (3*17 = 51 independent normalisations)

        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 48, temporal_kernel_size=temporal_kernel_size, residual=False),
            STGCNBlock(48, 48, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(48, 48, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(48, 48, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(48, 96, stride=2, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(96, 96, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(96, 96, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(96, 192, stride=2, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(192, 192, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
            STGCNBlock(192, 192, temporal_kernel_size=temporal_kernel_size, dropout=dropout),
        ])

        if edge_importance_weighting: # learn weights for each connection in the skeleton graph
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(self.adjacency)) for _ in self.stgcn_blocks
            ])
        else:
            self.edge_importance = [1.0 for _ in self.stgcn_blocks]

        self.classifier = nn.Conv2d(in_channels=192, out_channels=2, kernel_size=1)

    def forward(self, x):
        '''
        x shape: [B, C, T, V, M]
        where:
            B = Batch size
            C = input channels -> (X, Y, Confidence)
            T = Frames -> 150
            V = Joints -> 17
            M = people
        '''

        B, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 2, 1).contiguous()
        x = x.view(B*M, V*C, T)
        x = self.data_batch_norm(x)
        x = x.view(B, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B*M, C, T, V)

        for block, importance in zip(self.stgcn_blocks, self.edge_importance):
            weighted_adjacency = self.adjacency * importance 
            x = block(x, weighted_adjacency)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(B, M, 192, 1, 1)
        x = x.mean(dim=1)

        x = self.classifier(x)
        x = x.view(B, -1)

        return x