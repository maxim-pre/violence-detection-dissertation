import torch 
import torch.nn as nn 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class BaselineCNNLSTM(nn.Module):

    def __init__(
            self,
            hidden_size = 256, 
            num_layers = 1,
            num_classes = 2,
            dropout = 0.3, 
            freeze_cnn = True
    ):
        super().__init__()

        # Load MobileNetV2 with pretrained weights
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)

        # Extract the feature extractor (all layers except the final classifier)
        self.cnn = mobilenet.features

        # convert (1280, 7, 7) to (1280, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = 1280

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_size, num_classes)
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

        # global average pooling
        features = self.avgpool(features)
        # shape: (B*T, 1280, 1, 1)

        features = features.view(B, T, self.feature_dim)
        # shape: (B, T, 1280)

        lstm_out, (hidden, cell) = self.lstm(features)

        # use the last hidden state for classification
        final_hidden = hidden[-1]

        # class scores
        logits = self.classifier(final_hidden)

        return logits





    

        