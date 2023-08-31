import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
import os

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import csv

import scipy.stats as ss

from sklearn import svm, datasets
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F


from QNPy.CNP_ARCHITECTURE import DeterministicModel
from QNPy.CNP_METRICS import LogProbLoss, MSELoss, MAELoss
from QNPy.CNP_DATASETCLASS import LighCurvesDataset, collate_lcs

def create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/'):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)


def split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER):
    i=0
    for file in tqdm(files):

        lcName = file.split(".")[0]

        tmpDataFrame = pd.read_csv(os.path.join(DATA_SRC, file))

        r = random.uniform(0, 1)
        if r < 0.8:
            filename = TRAIN_FOLDER + lcName + '_split' + str(i) + '.csv'
        elif r < 0.9:
            filename = TEST_FOLDER + lcName + '_split' + str(i) + '.csv'
        else:
            filename = VAL_FOLDER + lcName + '_split' + str(i) + '.csv'

        tmpDataFrame.to_csv(filename, index=False)

        i=i+1

torch.cuda.empty_cache() 

# REPRODUCIBILITY  
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_data_loaders(data_path_train, data_path_val, batch_size):
    train_set = LighCurvesDataset(root_dir=data_path_train, status='train')
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_lcs,
                              num_workers=0,
                              pin_memory=True)

    val_set = LighCurvesDataset(root_dir=data_path_val, status='test')
    val_loader = DataLoader(val_set,
                            num_workers=0,
                            batch_size=1,
                            pin_memory=True)

    return train_loader, val_loader


def create_model_and_optimizer(device):
    model = DeterministicModel()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LogProbLoss()
    mseMetric = MSELoss()
    maeMetric = MAELoss()
    
    return model, optimizer, criterion, mseMetric, maeMetric

def train_model(model, trainLoader, valLoader, criterion, optimizer, num_runs, EPOCHS, EARLY_STOPPING_LIMIT, mseMetric, maeMetric, device):
    history_loss_train = [[] for _ in range(num_runs)]
    history_loss_val = [[] for _ in range(num_runs)]
    history_mse_train = [[] for _ in range(num_runs)]
    history_mse_val = [[] for _ in range(num_runs)]
    history_mae_train = [[] for _ in range(num_runs)]
    history_mae_val = [[] for _ in range(num_runs)]
    epoch_counter_train_loss = [[] for _ in range(num_runs)]
    epoch_counter_train_mse = [[] for _ in range(num_runs)]
    epoch_counter_train_mae = [[] for _ in range(num_runs)]
    epoch_counter_val_loss = [[] for _ in range(num_runs)]
    epoch_counter_val_mse = [[] for _ in range(num_runs)]
    epoch_counter_val_mae = [[] for _ in range(num_runs)]

    for j in range(num_runs):
        epochs_since_last_improvement = 0
        best_loss = None
        best_model = model.state_dict()
        epoch_counter = 0

        for epoch in tqdm(range(EPOCHS)):
            epoch_counter = epoch + 1
            model.train()
            total_loss_train = 0
            total_mse_train = 0
            total_mae_train = 0
            for data in trainLoader:
                # Unpack data
                [context_x, context_y, target_x, measurement_error], target_y = data

                # Move to GPU
                context_x, context_y, target_x, target_y, measurement_error = (
                    context_x.to(device),
                    context_y.to(device),
                    target_x.to(device),
                    target_y.to(device),
                    measurement_error.to(device),
                )

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                dist, mu, sigma = model(context_x, context_y, target_x)

                # Calculate loss and do a backward pass
                loss = criterion(dist, target_y)
                total_loss_train += loss.item()
                loss.backward()

                # Update weights
                optimizer.step()

                # Calculate MSE metric
                mseLoss = mseMetric(target_y, mu, measurement_error)
                total_mse_train += mseLoss

                # Calculate MAE metric
                maeLoss = maeMetric(target_y, mu, measurement_error)
                total_mae_train += maeLoss

            # Update history for losses
            epoch_loss = total_loss_train / len(trainLoader)
            history_loss_train[j].append(epoch_loss)

            epoch_mse = total_mse_train / len(trainLoader)
            history_mse_train[j].append(epoch_mse)

            epoch_mae = total_mae_train / len(trainLoader)
            history_mae_train[j].append(epoch_mae)

            # Validation
            model.eval()
            with torch.no_grad():
                total_loss_val = 0
                total_mse_val = 0
                total_mae_val = 0
                for data in valLoader:
                    # Unpack data
                    context_x, context_y, target_x, target_y, target_test_x, measurement_error = data[
                        "context_x"
                    ], data["context_y"], data["target_x"], data["target_y"], data["target_test_x"], data[
                        "measurement_error"
                    ]

                    # Move to GPU
                    context_x, context_y, target_x, target_y, target_test_x, measurement_error = (
                        context_x.to(device),
                        context_y.to(device),
                        target_x.to(device),
                        target_y.to(device),
                        target_test_x.to(device),
                        measurement_error.to(device),
                    )

                    # Forward pass
                    dist, mu, sigma = model(context_x, context_y, target_x)

                    # Calculate loss
                    loss = criterion(dist, target_y)
                    total_loss_val += loss.item()

                    # Calculate MSE metric
                    mseLoss = mseMetric(target_y, mu, measurement_error)
                    total_mse_val += mseLoss

                    # Calculate MAE metric
                    maeLoss = maeMetric(target_y, mu, measurement_error)
                    total_mae_val += maeLoss

            # Update history for losses
            val_loss = total_loss_val / len(valLoader)
            history_loss_val[j].append(val_loss)

            val_mse = total_mse_val / len(valLoader)
            history_mse_val[j].append(val_mse)

            val_mae = total_mae_val / len(valLoader)
            history_mae_val[j].append(val_mae)

            # Early stopping
            if best_loss is None:
                best_loss = val_loss

            if val_loss >= best_loss:
                epochs_since_last_improvement += 1
                if epochs_since_last_improvement >= EARLY_STOPPING_LIMIT:
                    print(f"Early stopped at epoch {epoch}!")
                    print(f"Best model at epoch {epoch - epochs_since_last_improvement}!")
                    model.load_state_dict(best_model)
                    break
            else:
                epochs_since_last_improvement = 0
                best_loss = val_loss
                best_model = model.state_dict()

            epoch_counter_train_loss[j].append(epoch_counter)
            epoch_counter_train_mse[j].append(epoch_counter)
            epoch_counter_train_mae[j].append(epoch_counter)
            epoch_counter_val_loss[j].append(epoch_counter)
            epoch_counter_val_mse[j].append(epoch_counter)
            epoch_counter_val_mae[j].append(epoch_counter)

    return (
        history_loss_train,
        history_loss_val,
        history_mse_train,
        history_mse_val,
        history_mae_train,
        history_mae_val,
        epoch_counter_train_loss,
        epoch_counter_train_mse,
        epoch_counter_train_mae,
        epoch_counter_val_loss,
        epoch_counter_val_mse,
        epoch_counter_val_mae,
    )

def save_lists_to_csv(file_names, lists):
    for file_name, data_list in zip(file_names, lists):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data_list:
                writer.writerow(row)


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file):
    history_loss_train = np.loadtxt(history_loss_train_file, delimiter=',')
    history_loss_val = np.loadtxt(history_loss_val_file, delimiter=',')
    epoch_counter_train_loss = np.loadtxt(epoch_counter_train_loss_file, delimiter=',')
    
    history_loss_train = history_loss_train[:len(epoch_counter_train_loss)]
    history_loss_val = history_loss_val[:len(epoch_counter_train_loss)]

   
    plt.plot(epoch_counter_train_loss, history_loss_train, label='Train LOSS')
    plt.plot(epoch_counter_train_loss, history_loss_val, label='Validation LOSS')
    plt.title("LosProbLOSS")
    plt.xlabel("Epoch")
    plt.ylabel("LOSS")
    plt.legend()
    plt.show()

def plot_mse(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file):
    history_mse_train = np.loadtxt(history_mse_train_file, delimiter=',')
    history_mse_val = np.loadtxt(history_mse_val_file, delimiter=',')
    epoch_counter_train_mse = np.loadtxt(epoch_counter_train_mse_file, delimiter=',')

    history_mse_train = history_mse_train[:len(epoch_counter_train_mse)]
    history_mse_val = history_mse_val[:len(epoch_counter_train_mse)]

    epoch_counter = len(epoch_counter_train_mse)
    plt.plot(epoch_counter_train_mse, history_mse_train, label='Train MSE')
    plt.plot(epoch_counter_train_mse, history_mse_val, label='Validation MSE')
    plt.title("Mean Squared Error (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_mae(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file):
    history_mae_train = np.loadtxt(history_mae_train_file, delimiter=',')
    history_mae_val = np.loadtxt(history_mae_val_file, delimiter=',')
    epoch_counter_train_mae = np.loadtxt(epoch_counter_train_mae_file, delimiter=',')

    history_mae_train = history_mae_train[:len(epoch_counter_train_mae)]
    history_mae_val = history_mae_val[:len(epoch_counter_train_mae)]

    epoch_counter = len(epoch_counter_train_mae)
    plt.plot(epoch_counter_train_mae, history_mae_train, label='Train MAE')
    plt.plot(epoch_counter_train_mae, history_mae_val, label='Validation MAE')
    plt.title("Mean Absolute Error (MAE)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()


def save_model(model, MODEL_PATH):
    torch.save(model.state_dict(), MODEL_PATH)
