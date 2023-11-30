import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load
import pathlib
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    


class ECG_CNN_TransformerBIN(nn.Module):
    def __init__(self):
        super(ECG_CNN_TransformerBIN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1,)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(256*800 + 12*800 , 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
        self.drop5 = nn.Dropout(0.8)

    def forward(self, x):
        cnn_x = self.relu(self.conv1(x))
        cnn_x = self.relu(self.conv2(cnn_x))
        cnn_x = self.relu(self.conv3(cnn_x))
        cnn_x = cnn_x.view(x.size(0), -1)

        #cnn_xf = self.relu(self.conv1(full_x))
        #cnn_xf = cnn_xf.view(full_x.size(0), -1)
        #cnn_xf = self.drop5(cnn_xf)

        x = x.permute(2, 0, 1)
        transformer_x = self.transformer_encoder(x)
        transformer_x = transformer_x.permute(1, 0, 2)
        transformer_x = transformer_x.reshape(transformer_x.size(0), -1)
        #transformer_x = self.drop(transformer_x) #drop показал себя хуже

        #Конкатинируем результаты cnn и transformer слоёв
        x = torch.cat((cnn_x, transformer_x), dim=1)

        x = self.relu(self.fc1(x))
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.sig(x)
        return x

class ECG_CNN_Transformer(nn.Module):
    def __init__(self):
        super(ECG_CNN_Transformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1,)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(256*700 + 12*700 , 128)
        self.fc2 = nn.Linear(128, 6)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
        self.drop5 = nn.Dropout(0.8)

    def forward(self, x):
        cnn_x = self.relu(self.conv1(x))
        cnn_x = self.relu(self.conv2(cnn_x))
        cnn_x = self.relu(self.conv3(cnn_x))
        cnn_x = cnn_x.view(x.size(0), -1)
        
        #cnn_xf = self.relu(self.conv1(full_x))
        #cnn_xf = cnn_xf.view(full_x.size(0), -1) 
        #cnn_xf = self.drop5(cnn_xf)

        x = x.permute(2, 0, 1)  
        transformer_x = self.transformer_encoder(x)
        transformer_x = transformer_x.permute(1, 0, 2)  
        transformer_x = transformer_x.reshape(transformer_x.size(0), -1) 
        #transformer_x = self.drop(transformer_x)

        x = torch.cat((cnn_x, transformer_x), dim=1)

        x = self.relu(self.fc1(x))
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.sig(x)
        return x
