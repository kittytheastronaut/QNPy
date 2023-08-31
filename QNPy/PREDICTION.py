import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import collections
import numpy as np
import math

import random
import csv
from datetime import datetime

from cycler import cycler

import os
import glob

from tqdm import tqdm

import json

import pandas as pd

from sklearn import svm, datasets
import scipy.stats as ss

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error

from QNPy.CNP_ARCHITECTURE import DeterministicModel
from QNPy.CNP_METRICS import LogProbLoss, MSELoss
from QNPy.CNP_DATASETCLASS import LighCurvesDataset, collate_lcs

def prepare_output_dir(OUTPUT_PATH):
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for name in files:
            os.remove(os.path.join(root, name))
            


def load_trained_model(MODEL_PATH, device):
    model = DeterministicModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()
    
    return model



def get_criteria():
    criterion = LogProbLoss()
    mseMetric = MSELoss()
    
    return criterion, mseMetric


OUTPUT_PATH="./output/predictions/"


def plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save=False, flagval=0, isTrainData=None, notTrainData=None):
    # Move to cpu
    target_x, target_y, context_x, context_y, pred_y, var = target_x.cpu(), target_y.cpu(), \
                                                              context_x.cpu(), context_y.cpu(), \
                                                              pred_y.cpu(), var.cpu()

    target_test_x = target_test_x.cpu()

    # Plot everything
    plt.plot(target_test_x[0], pred_y[0], 'b-', linewidth=1.5, label='mean model')
    plt.plot(target_x[0], target_y[0], linestyle='', linewidth=1.3, color='k')
    plt.plot(context_x[0], context_y[0], marker='|', linestyle='', linewidth=1.3, color='k', label='observations')

    plt.fill_between(
        target_test_x[0, :],
        pred_y[0, :] - var[0, :],
        pred_y[0, :] + var[0, :],
        alpha=0.2,
        facecolor='#ff9999',
        interpolate=True)

    plt.legend()

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title(lcName)

    if save:
        if isTrainData and flagval == 0:
            savePath = os.path.join(OUTPUT_PATH, 'train')
        elif notTrainData and flagval == 1:
            savePath = os.path.join(OUTPUT_PATH, 'val')
        else:
            savePath = os.path.join(OUTPUT_PATH, 'test')

        lcName = lcName.split(',')[0]
        pltPath = os.path.join(savePath, 'plots', lcName + '.png')
        csvpath = os.path.join(savePath, 'data', lcName + '_predictions.csv')

        if not os.path.exists(os.path.join(savePath, 'plots')):
            os.makedirs(os.path.join(savePath, 'plots'))

        if not os.path.exists(os.path.join(savePath, 'data')):
            os.makedirs(os.path.join(savePath, 'data'))

        plt.savefig(pltPath, bbox_inches='tight')
        plt.clf()

        # Create dataframe with predictions and save csv
        d = {'time': target_test_x[0], 'cont': pred_y[0], 'conterr': var[0]}
        df = pd.DataFrame(data=d)

        df.to_csv(csvpath, index=False)

        if not os.path.exists(pltPath):
            print(pltPath)
    else:
        plt.show()


def load_test_data(data_path):
    testSet = LighCurvesDataset(root_dir = data_path, status = 'test')
    testLoader = DataLoader(testSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return testLoader


def load_train_data(data_path):
    train_set = LighCurvesDataset(root_dir=data_path, status='test')
    train_loader = DataLoader(train_set, 
                              num_workers=0, 
                              batch_size=1, 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader

def load_val_data(data_path):
    valSet = LighCurvesDataset(root_dir = data_path, status = 'test')
    valLoader = DataLoader(valSet,
                           num_workers = 0,
                           batch_size  = 1, 
                           pin_memory  = True)
    return valLoader

def plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device):
    testMetrics = {}

    with torch.no_grad():
        for data in tqdm(testLoader):
            # Unpack data
            lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], \
                                                                              data['target_y'], data['target_test_x'], data['measurement_error']

            # Move to gpu
            context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                      target_x.to(device), target_y.to(device), \
                                                                      target_test_x.to(device), measurement_error.to(device)

            # Forward pass
            dist, mu, sigma = model(context_x, context_y, target_x)

            # Calculate loss
            loss = criterion(dist, target_y)
            loss = loss.item()

            # Calculate MSE metric
            mseLoss = mseMetric(target_y, mu, measurement_error)

            # Discard .csv part of LC name
            lcName = lcName[0].split('.')[0]

            # Add metrics to map
            testMetrics[lcName] = {'log_prob:': str(loss),
                                      'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)
            plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save = True, isTrainData = False, flagval =0)
    
    return testMetrics


def save_test_metrics(OUTPUT_PATH, testMetrics):
    with open(OUTPUT_PATH + 'test/testMetrics.json', 'w') as fp:
        json.dump(testMetrics, fp, indent=4)

def plot_light_curves_from_train_set(trainLoader, model, criterion, mseMetric, plot_function, device):
    # Plots all light curves from train set

    trainMetrics = {}

    counter = 0

    with torch.no_grad():
        for data in tqdm(trainLoader):
            # End after predicting given number of LCs
 #           if counter > 100:
 #               break
 #           counter += 1

            try:
                lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                          data['context_y'], data['target_x'], data['target_y'], \
                                                                          data['target_test_x'], data['measurement_error']

                # Move to gpu
                context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                          target_x.to(device), target_y.to(device), \
                                                                          target_test_x.to(device), measurement_error.to(device)

                # Forward pass
                dist, mu, sigma = model(context_x, context_y, target_x)

                # Calculate loss
                loss = criterion(dist, target_y)
                loss = loss.item()

                # Calculate MSE metric
                mseLoss = mseMetric(target_y, mu, measurement_error)

                # Discard .csv part of LC name
                lcName = lcName[0].split('.')[0]

                # Add metrics to map
                trainMetrics[lcName] = {'log_prob': str(loss),
                                        'mse': str(mseLoss)}

                # Add loss value to LC name
                lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

                # Predict and plot
                dist, mu, sigma = model(context_x, context_y, target_test_x)
                plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save = True, isTrainData = True)
            except Exception as e:
                print(f'Error in: {lcName}')
    return trainMetrics


def save_train_metrics(OUTPUT_PATH, trainMetrics):
    with open(OUTPUT_PATH + 'train/trainMetrics.json', 'w') as fp:
        json.dump(trainMetrics, fp, indent=4)

def plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device):
    valMetrics = {}

    counter = 0

    with torch.no_grad():
        for data in tqdm(valLoader):
            # End after predicting given number of LCs
            if counter > 100:
                break
            counter += 1

            try:
                lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], data['target_y'], \
                                                                              data['target_test_x'], data['measurement_error']

                # Move to GPU
                context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                              target_x.to(device), target_y.to(device), \
                                                                              target_test_x.to(device), measurement_error.to(device)

                # Forward pass
                dist, mu, sigma = model(context_x, context_y, target_x)

                # Calculate loss
                loss = criterion(dist, target_y)
                loss = loss.item()

                # Calculate MSE metric
                mseLoss = mseMetric(target_y, mu, measurement_error)

                # Discard .csv part of LC name
                lcName = lcName[0].split('.')[0]

                # Add metrics to map
                valMetrics[lcName] = {'log_prob': str(loss),
                                        'mse': str(mseLoss)}

                # Add loss value to LC name
                lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

                # Predict and plot
                dist, mu, sigma = model(context_x, context_y, target_test_x)
                plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save=True, isTrainData=False, flagval=1, notTrainData=True)
            except Exception as e:
                print(f'Error in: {lcName}')
    return valMetrics


def save_val_metrics(OUTPUT_PATH, valMetrics):
    with open(OUTPUT_PATH + 'val/valMetrics.json', 'w') as fp:
        json.dump(valMetrics, fp, indent=4)
