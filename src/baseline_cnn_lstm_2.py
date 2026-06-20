import torch 
import torch.nn as nn 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sepconv_lstm import SepConvLSTM


# this model utilises sepConvLSTM
class BaselineCNNLSTM2(nn.Module):

    def __init__(
            self,
            hidden_channels = 128, 
            num_classes = 2,
            dropout = 0.4, 
            freeze_cnn = True,
            partial_freeze_cnn = False
    ):
        super().__init__()

        # Load MobileNetV2 with pretrained weights
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)

        # Extract the feature extractor (all layers except the final classifier)
        self.cnn = mobilenet.features

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if partial_freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
            for param in self.cnn[14:].parameters():
                param.requires_grad = True

        
        self.feature_dim = 1280

        self.channel_reduce = nn.Conv2d(
            in_channels=self.feature_dim,
            out_channels=128,
            kernel_size=1
        )

        self.sepconvlstm = SepConvLSTM(
            input_channels=128,
            hidden_channels=hidden_channels,
            kernel_size=3
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x):
        '''
        x shape: (B, T, C, H, W)

        B = batch size
        T = number of frames
        C = channels
        H = height
        W = width

        '''

        B, T, C, H, W = x.shape

        # for the cnn feature extracter we need to treat all frames as individual images
        x = x.view(B * T, C, H, W)

        # extract CNN feature maps
        features = self.cnn(x)
        # shape: (B*T, 1280, 7, 7)

        features = self.channel_reduce(features)
        # shape: (B*T, 128, 7, 7)

        _, C_feat, H_feat, W_feat = features.shape

        # restore video structure
        features = features.view(B, T, C_feat, H_feat, W_feat)
        # shape: (B, T, 128, 7, 7)

        outputs = self.sepconvlstm(features)
        # shape: (B, T, hidden_channels, 7, 7)

        last_hidden = outputs[:, -1]
        # shape: (B, hidden_channels, 7, 7)

        pooled = self.avgpool(last_hidden)
        # shape: (B, hidden_channels, 1, 1)

        pooled = torch.flatten(pooled, start_dim=1)
        # shape: (B, hidden_channels)

        logits = self.classifier(pooled)
        # shape: (B, 2)

        return logits





    

        