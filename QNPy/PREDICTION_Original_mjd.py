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

import dill

import math

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

def load_trcoeff():
    with open("trcoeff.pickle", "rb") as f:
        tr = dill.load(f)
    return tr

def back_x(x,Ax,Bx):
  a=-2
  b=2
  gornji=Bx-Ax
  donji=b-a
  xorig=Ax+((x-a)*(gornji/donji))
  return xorig

def back_y(y,Ay,By):
  a=-2
  b=2
  gornji=By-Ay
  donji=b-a
  yorig=Ay+((y-a)*(gornji/donji))
  return yorig

OUTPUT_PATH="./output/predictions/"
    
###NEW PLOT FUNCTION
def plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, isTrainData = None, flagval = 0, notTrainData = None):
    """Plots the light curve data and predicted mean and variance.

    Args: 
    context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    yerr1: Array of shape BATCH_SIZE x NUM_measurement_error that contains the measurement errors.
    pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    tr: array of data in pickle format needed to backtransform data from [-2,2] x [-2,2] to MJD x Mag
    """
    # Move to cpu
    target_x, target_y, context_x, context_y,yerr1, pred_y, var = target_x.cpu(), target_y.cpu(), \
                                                              context_x.cpu(), context_y.cpu(), yerr1.cpu(),\
                                                              pred_y.cpu(), var.cpu()

    target_test_x = target_test_x.cpu()
        
    target_test_xorig=back_x(target_test_x[0].numpy(),tr[0][1],tr[0][2])
    
    pred_yorig=back_y(pred_y[0].numpy(),tr[0][3],tr[0][4])

    target_yorig=back_y(target_y[0].numpy(),tr[0][3],tr[0][4])
    target_xorig=back_x(target_x[0].numpy(),tr[0][1],tr[0][2])
    varorig=back_y(var[0, :].numpy(),tr[0][3],tr[0][4])
    # Plot everything
    plt.plot(target_test_xorig, pred_yorig, 'b-', linewidth=1.5, label = 'mean model')
    plt.errorbar(target_xorig, target_yorig, yerr=yerr1[0], linestyle='', elinewidth=1.3, color='k',label = 'observations')
   
    
    plt.fill_between(
      target_test_xorig,
      pred_yorig - var[0, :].numpy(),
      pred_yorig + var[0, :].numpy(),
      alpha=0.6,
      facecolor='#ff9999', label='CI',
      interpolate=True)

    
    plt.legend(fontsize=10)
    minx=math.ceil(target_test_xorig.min())
    maxx=math.ceil(target_test_xorig.max())
    middlex=math.ceil((minx+maxx)/2.)
    miny=round( target_yorig.min()-yerr1.numpy().max(),1)
    maxy=round( target_yorig.max()+yerr1.numpy().max(),1)
    middley=round((miny+maxy)/2.)

    # Make the plot pretty
    plt.yticks([miny, middley, maxy])
    plt.xticks([minx, middlex, maxx])

    plt.tick_params(axis='both', which='minor', labelsize=8.5)
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title(lcName, fontsize=8)

    
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
        d = {'time': target_test_x[0], 
             'cont': pred_y[0],
             'conterr': var[0]}
        
        df = pd.DataFrame(data=d)
        
        df.to_csv(csvpath, index=False)
        
        if os.path.exists(pltPath + ".png") == False:
            print(pltPath)
    else:
        
        print('')

def load_test_data(DATA_PATH_TEST):
    testSet = LighCurvesDataset(root_dir = DATA_PATH_TEST, status = 'test')
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

def find_LC_transform(lists, search):
  search1=search
  return list(filter(lambda x:x[0]==search1,lists))

import torch
from tqdm import tqdm

def plot_test_data(model, testLoader, criterion, mseMetric, plot_function2, device,tr):
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

            #Take name for finding coefficinetns for backward propagation 
            llc = lcName                                                                 

            # Add metrics to map
            testMetrics[lcName] = {'log_prob:': str(loss),
                                      'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)

            #coeeficinets for transformation back
            ZX = find_LC_transform(tr, llc[:7])

            plot_function2(tr, target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, save=True, isTrainData=False, flagval=0)

    return testMetrics


def save_test_metrics(OUTPUT_PATH, testMetrics):
    with open(OUTPUT_PATH + 'test/testMetrics.json', 'w') as fp:
        json.dump(testMetrics, fp, indent=4)

def plot_train_light_curves(trainLoader, model, criterion, mseMetric, plot_function, device, tr):
    """
    Plots light curves from test set in original range MJD x Mag

    Args:
        model (torch.nn.Module): Trained probabilistic model
        criterion (torch.nn.Module): Loss function
        mseMetric (function): Mean squared error metric
        trainLoader (torch.utils.data.DataLoader): Dataloader for training set
        device (str): Device for PyTorch model
        tr (pd.DataFrame): Transforming coefficients DataFrame for LCs

    Returns:
        trainMetrics (dict): Log probability and MSE metrics for train set

    """
    trainMetrics = {}
    counter = 0

    with torch.no_grad():

        for data in tqdm(trainLoader):
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

            # Take name for finding coefficients for backward propagation
            llc = lcName

            # Add metrics to map
            trainMetrics[lcName] = {'log_prob:': str(loss),
                                      'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)
            # Coefficients for transformation back
            ZX = find_LC_transform(tr,llc[:7])

            plot_function2(tr, target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, save=True, isTrainData=True, flagval=0)

    return trainMetrics


def save_train_metrics(OUTPUT_PATH, trainMetrics):
    with open(OUTPUT_PATH + 'train/trainMetrics.json', 'w') as fp:
        json.dump(trainMetrics, fp, indent=4)

import torch
from tqdm import tqdm

def plot_val_curves(model, valLoader, criterion, mseMetric, plot_function, device, tr):
    # train_loss = 0
    valMetrics = {}
    counter = 0

    with torch.no_grad():
        for data in tqdm(valLoader):
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

            #Take name for finding coefficinetns for backward propagation 
            llc=lcName                                                                 

            # Add metrics to map
            valMetrics[lcName] = {'log_prob:': str(loss),
                                      'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)
            #coeeficinets for transformation back
            ZX=find_LC_transform(tr,llc[:7])

            plot_function2(tr,target_x, target_y, context_x, context_y, measurement_error,mu, sigma, target_test_x, lcName, save = True, isTrainData = False, notTrainData = True, flagval = 1)

    #train_loss = train_loss / len(trainLoader)
    return valMetrics


def save_val_metrics(OUTPUT_PATH, valMetrics):
    with open(OUTPUT_PATH + 'val/valMetrics.json', 'w') as fp:
        json.dump(valMetrics, fp, indent=4)
