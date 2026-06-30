import torch 
import torch.nn as nn 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .sepconv_lstm import SepConvLSTM


# this model utilises sepConvLSTM
class CNNLSTMV2(nn.Module):

    def __init__(
            self,
            hidden_channels = 128, 
            reduced_channels = 64,
            classifier_hidden_size = 128,
            num_classes = 2,
            dropout = 0.4, 
            cnn_cutoff = 19,
            cnn_unfreeze_from=None
    ):
        super().__init__()

        # Load MobileNetV2 with pretrained weights
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)


        # Truncate MobileNet feature extractor
        self.cnn = nn.Sequential(*mobilenet.features[:cnn_cutoff])

        if cnn_unfreeze_from is None:
            for param in self.cnn.parameters():
                param.requires_grad = False

        else:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
            for param in self.cnn[cnn_unfreeze_from:].parameters():
                param.requires_grad = True

        # Infer output channels after truncation
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.cnn(dummy)
            feature_dim = out.shape[1]
        
        self.channel_reduce = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=reduced_channels,
            kernel_size=1
        )

        self.sepconvlstm = SepConvLSTM(
            input_channels=reduced_channels,
            hidden_channels=hidden_channels,
            kernel_size=3
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_channels, classifier_hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes)
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
        # shape: (B*T, feature_dim, 7, 7)

        features = self.channel_reduce(features)
        # shape: (B*T, reduced_channels, 7, 7)

        _, C_feat, H_feat, W_feat = features.shape

        # restore video structure
        features = features.view(B, T, C_feat, H_feat, W_feat)
        # shape: (B, T, 128, 7, 7)

        outputs = self.sepconvlstm(features)
        # shape: (B, T, hidden_channels, 7, 7)

        last_hidden = outputs[:, -1]
        # shape: (B, hidden_channels, 7, 7)

        last_hidden = self.maxpool(last_hidden)
        # (B, hidden_channels, 3, 3)

        pooled = self.avgpool(last_hidden)
        # shape: (B, hidden_channels, 1, 1)

        pooled = torch.flatten(pooled, start_dim=1)
        # shape: (B, hidden_channels)

        logits = self.classifier(pooled)
        # shape: (B, 2)

        return logits





    

        