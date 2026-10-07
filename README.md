# Explainable AI for Violence Detection

This project evaluates explainable artificial intelligence (XAI) techniques for video-based violence detection. Two deep learning models were developed using different input representations: a CNN-LSTM operating on RGB frame differences and an ST-GCN operating on estimated human skeletal keypoints. Gradient- and perturbation-based XAI techniques were then implemented and evaluated using measures of faithfulness, stability, and computational efficiency.
![overview](static/overview.png)

## Dataset

The project uses the [**RWF-2000**](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) dataset, a real-world video dataset for violence detection collected entirely from CCTV footage.

- **2,000 videos** in total
- **1,000 Fight** and **1,000 Non-Fight** videos
- **1,600 training videos** (800 Fight / 800 Non-Fight)
- **400 validation videos** (200 Fight / 200 Non-Fight)
- Each video is **5 seconds long**
- Each video contains **150 frames at 30 FPS**
- Videos contain a range of scenes, viewpoints, lighting conditions, and resolutions

<p align="center">
  <img src="static/crowded.gif" width="49%">
  <img src="static/transient.gif" width="49%">
</p>


## Models

Two models with fundamentally different input representations were implemented to evaluate XAI techniques across both image-based and comparatively underexplored skeleton-based violence detection. 

### CNN-LSTM

The CNN-LSTM detects violence from sequences of RGB frame differences. The pre-trained MobileNetV2 CNN backbone extracts spatial feature maps from each frame in the clip, which are then fed sequentially into a SepConvLSTM to extract spatiotemporal features before classification. 

1. 33 frames are uniformly sampled from the original 150-frame video. As consecutive frames typically contain redundant information (as they are very similar), temporal sampling reduces the amount of data processed while retaining information across the full video.

2. The frames were resized to 224 X 224 and augmentations including random cropping, horizontal flipping, random contrast, random brightness, and Gaussian blurring were applied during training to improve generalisation. 

3. The 33 sampled frames are differenced to obtain 32 frame differences. Frame differencing highlights motion between consecutive frames while suppressing static background information.

4. The 32 frame differences are passed independently through a pre-trained MobileNetV2 CNN backbone to extract a spatial feature map from each frame difference.

5. The spatial feature maps are fed sequentially into a SepConvLSTM to extract spatiotemporal features. Unlike a conventional LSTM, it uses convolutional operations, which preserve the spatial structure of the feature maps. Depthwise separable convolutions reduce parameter count and computational cost.

6. The final SepConvLSTM output is pooled and passed through a fully connected classification layer to produce the final Fight or Non-Fight prediction.

![Grad-CAM digram](static/cnn_lstm_diagram.png)

### ST-GCN
![ST-GCN diagram](static/stgcn_diagram.png)

## Explainable AI Methods



