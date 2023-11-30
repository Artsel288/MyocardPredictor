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
    

def plot_data(data, file, nrows=4, ncols=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 7))

    for idx, ax in enumerate(axes.flatten()):
        if idx < len(data):
            ax.plot(data[idx])
        
    fig.savefig(f'static/plots/{file.replace("npy", "jpg")}')
    plt.close(fig)
    

def inference_ovBIN(model, x, overlap=0.875):
    num_windows = int((x.shape[1] * overlap) // 800) + (1 if (x.shape[1] * overlap) % 800 != 0 else 0)
    step_size = int(800 * overlap)
    preds = []
        
    with torch.no_grad():
        model.eval()
        
        for i in range(num_windows):
            start = i * step_size
            end = start + 800
            x_window = x[:, start:end]
            pred = model(x_window.unsqueeze(0))
            preds.append(pred)
        
        avg_pred = sum(preds) / num_windows
        
    return avg_pred


def inference_ov(model, x, overlap=0.98):
    num_windows = int((x.shape[1] * overlap) // 700) + (1 if (x.shape[1] * overlap) % 700 != 0 else 0)
    step_size = int(700 * overlap)
    preds = []
    
    with torch.no_grad():
        model.eval()
        
        for i in range(num_windows):
            start = i * step_size
            end = start + 700
            x_window = x[:, start:end]
            pred = model(x_window.unsqueeze(0))
            preds.append(pred)
        
        avg_pred = sum(preds) / num_windows
        
    return avg_pred

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
