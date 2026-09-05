# Explainable AI for Violence Detection

Source code for my MSc AI dissertation at the University of Bath.

This project investigates explainable artificial intelligence (XAI) techniques for video-based violence detection. Two models were implemented

- a CNN-LSTM operating on frame differences;
- a skeleton-based ST-GCN operating on skeletal data.

## Project Structure

```text

├── CNN_LSTM_final_model/       # Final CNN-LSTM model
├── STGCN_final_model/          # Final ST-GCN model
│
├── src/
│   ├── cnn_lstm/               # CNN-LSTM model and training code
│   ├── Skeleton_model/         # ST-GCN model and training code
│   ├── XAI/                    # XAI techniques and evaluation
│   ├── config.py               
│   └── rwf2000.py              # RWF-2000 dataset class for both models
│
├── scripts/
│   ├── cnn_lstm/               # CNN-LSTM training script
│   ├── skeleton_model/         # Building pose DB + training script
│   └── common/                 # Shared utilities and XAI scripts
│
├── requirements.txt
└── README.md
```
## Data

The dataset is not included in this repository. It can be downloaded from Kaggle:

RWF-2000 Dataset: https://www.kaggle.com/datasets/vulamnguyen/rwf2000

## References

The implementations in this project were developed with reference to the original papers and associated repositories where applicable.

- **ST-GCN:** Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018.  
  Repository: https://github.com/yysijie/st-gcn

- **Grad-CAM:** Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.  


- **RISE:** Petsiuk, V., Das, A. and Saenko, K., 2018. RISE: Randomized Input Sampling for Explanation
of Black-box Models.  

- **SepConvLSTM:** based on the architecture described in: Islam, Z., Rukonuzzaman, M., Ahmed, R., Kabir, M.H. and Farazi, M., 2021. Efficient Two-
Stream Network for Violence Detection Using Separable Convolutional LSTM  

Additional academic references relating to the models, XAI techniques and evaluation methods are provided in the dissertation.