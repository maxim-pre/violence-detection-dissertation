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

class MaskedBatchNorm1d(nn.BatchNorm1d):
    # computes batch norm statistics from just the non-zero padded tracks

    def forward(self, x, mask=None):
        # x: [B*M, V*C, T]
        # mask: [B*M]

        if mask is None or not self.training: # dont update statistics if not training
            return super().forward(x)

        real = x[mask]
        batch_mean = real.mean(dim=(0,2))
        batch_var = real.var(dim=(0,2), unbiased=False)

        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        # set training to false because we've already computed running mean/var
        return F.batch_norm(x, batch_mean, batch_var, self.weight, self.bias, training=False, eps=self.eps)

class MaskedBatchNorm2d(nn.BatchNorm2d):
    # computes batch norm statistics from just the non-zero padded tracks

    def forward(self, x, mask=None):
        # x: [B*M, C, T, V]
        # mask: [B*M]

        if mask is None or not self.training: # dont update statistics if not training
            return super().forward(x)

        real = x[mask]
        batch_mean = real.mean(dim=(0,2,3))
        batch_var = real.var(dim=(0,2,3), unbiased=False)

        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        # can't use F.batch_norm with 2d for some reason
        mean = batch_mean.view(1, -1, 1, 1)   # [1, C, 1, 1]
        var = batch_var.view(1, -1, 1, 1)     
        weight = self.weight.view(1, -1, 1, 1)  
        bias = self.bias.view(1, -1, 1, 1)      

        std = torch.sqrt(var + self.eps) 
        normalised_x = (x - mean) / std # [B*M, C, T, V]
        normalised_x = normalised_x * weight + bias  

        return normalised_x

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions=3):
        
        # in_channels: number of channels in the input data 
        # out_channels: number of channels produced by the convolution
        # num_partitions = 3 (root, centripetal, centrifugal)
        
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
        # x shape [B, in_channels, 150, 17]
        # adjacency shape [3, 17, 17]
        # returns [B, out_channels, 150, 17]

        x = self.channel_transform(x) # [B, 3*out_channels, 150, 17]
        B, _, T, V = x.shape

        x = x.reshape(B, self.num_partitions, self.out_channels, T, V)

        # equation 10 in paper (sum feature maps from each partition)
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

        self.bn1 = MaskedBatchNorm2d(out_channels)

        self.temporal_conv = nn.Sequential(
            nn.ReLU(inplace=True), 
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1), 
                padding=(temporal_padding, 0),
                bias=False
            ),
        )

        self.bn2 = MaskedBatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = self._build_residual_connection(
            in_channels=in_channels, 
            out_channels=out_channels, 
            stride=stride, 
            residual=residual
        )

        self.relu = nn.ReLU(inplace=True)

    def _residual_fn(self, x, mask=None):
        x = self.residual_conv(x)
        x = self.residual_bn(x, mask=mask)
        return x


    def _build_residual_connection(self, in_channels, out_channels, stride, residual):
        if not residual:
            return lambda x, mask=None: 0
        elif (in_channels == out_channels) and (stride == 1): # tensor shape is the same so input can be added directly
            return lambda x, mask=None: x 
        else:
            self.residual_conv = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=(stride, 1), 
                bias=False
            )
            self.residual_bn = MaskedBatchNorm2d(out_channels)
            return self._residual_fn

    def forward(self, x, adjacency, mask=None):
        residual_output = self.residual(x, mask=mask)
        x = self.spatial_graph_conv(x, adjacency)

        x = self.bn1(x, mask=mask)
        x = self.temporal_conv(x)
        x = self.bn2(x, mask=mask)
        x = self.dropout(x)

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
        
        super().__init__()

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.data_batch_norm=MaskedBatchNorm1d(in_channels * num_joints)
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
        # x shape: [B, 3, 150, 17, 4]

        B, C, T, V, M = x.shape

        has_confidence = x[:, 2] > 0 # shape [B, T, V, M]
        present_in_frame = has_confidence.any(dim=2) # shape [B, T, M]
        presence = present_in_frame.any(dim=1).float() # shape [B, M]
        presence_flat = presence.view(B*M).bool()

        x = x.permute(0, 4, 3, 2, 1).contiguous()
        x = x.view(B*M, V*C, T)
        x = self.data_batch_norm(x, mask=presence_flat)
        x = x.view(B, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B*M, C, T, V)

        for block, importance in zip(self.stgcn_blocks, self.edge_importance):
            weighted_adjacency = self.adjacency * importance 
            x = block(x, weighted_adjacency, mask=presence_flat) 

        # x = [B*M, 256, 38, 17]
    
        x = F.avg_pool2d(x, x.size()[2:]) # [B*M, 256, 1, 1]
        x = x.view(B, M, 256)

        if self.people_aggregation == "masked_mean":
            mask = presence.unsqueeze(-1) # [B, M, 1]
            summed = (x * mask).sum(dim=1) # [B, 256]
            count = mask.sum(dim=1).clamp(min=1.0) # [B, 1]
            x = summed / count # [B, 256]

        elif self.people_aggregation == "max":
            # push absent people to -inf so they never win the max
            mask = presence.unsqueeze(-1).bool()                 # [B, M, 1]
            x = x.masked_fill(~mask, float("-inf"))
            x = x.max(dim=1).values                              # [B, 256]
            # max over all -inf gives -inf — clamp back to 0
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        else:
            raise ValueError("people_aggregation must be 'masked_mean' or 'max'")


        x = x.view(B, x.size(1), 1, 1) # [B, 256, 1, 1]
        x = self.classifier(x) # [B, 2, 1, 1]
        x = x.view(B, -1) # [B, 2]

        return x



# not used 
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