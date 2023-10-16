import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import collections
import numpy as np
import math
import matplotlib.backends.backend_pdf

import random
import csv
from datetime import datetime

from cycler import cycler

import os
import glob

from tqdm import tqdm

import json

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

def clear_output_dir(output_path):
    """
    Removes all files in the specified output directory.

    Args:
        output_path (str): The path to the output directory.
    """
    for root, dirs, files in os.walk(output_path):
        for name in files:
            os.remove(os.path.join(root, name))

def load_model(model_path, device):
    """
    Loads a trained model from disk and moves it to the specified device.

    Args:
        model_path (str): The path to the saved model.
        device (str or torch.device): The device to load the model onto.

    Returns:
        The loaded model.
    """
    model = DeterministicModel()  # replace with the appropriate model initialization code
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def get_criteria():
    criterion = LogProbLoss()
    mseMetric = MSELoss()
    
    return criterion, mseMetric

def remove_padded_values_and_filter(folder_path):
    # Get the list of CSV files in the input folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

    # Iterate over the CSV files
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)

        # Check if the filename contains "minus" or "plus"
        if "minus" in filename or "plus" in filename:
            # Delete the CSV file
            os.remove(file_path)
            print(f"Deleted file with 'minus' or 'plus' in the name: {filename}")
        elif "original" not in filename:
            # Delete the CSV file if it doesn't contain "original" in the name
            os.remove(file_path)
            print(f"Deleted file without 'original' in the name: {filename}")
        else:
            # Remove padded values from the curve and keep the original
            try:
                # Read the CSV file and load the data into a pandas DataFrame
                data = pd.read_csv(file_path)

                # Check if the DataFrame has more than 1 row
                if len(data) > 1:
                    # Check if 'mag' and 'magerr' columns are the same as the last observation
                    last_row = data.iloc[-1]
                    mag_values = data['cont']
                    magerr_values = data['conterr']
                    if not (mag_values == last_row['cont']).all() or not (magerr_values == last_row['conterr']).all():
                        # Keep rows where 'mag' and 'magerr' are not the same as the last observation
                        data = data[(mag_values != last_row['cont']) | (magerr_values != last_row['conterr'])]

                        # Overwrite the original CSV file with the modified DataFrame
                        data.to_csv(file_path, index=False)
                        print(f"Removed padding in file: {filename}")
                    else:
                        print(f"No padding removed for file: {filename}")
                else:
                    print(f"No padding removed for file: {filename}")

            except pd.errors.EmptyDataError:
                print(f"Error: Empty file encountered: {filename}")


OUTPUT_PATH="./output/predictions/"

####PLOT FUNCTION FOR TRANSFORMED DATA [-2,2]
def plot_function(target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, flagval =0, isTrainData = None, notTraindata=None):
    """Plots the light curve data and predicted mean and variance.

    Args: 
    context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    """
    # Move to cpu
    target_x, target_y, context_x, context_y,yerr1, pred_y, var = target_x.cpu(), target_y.cpu(), \
                                                              context_x.cpu(), context_y.cpu(), yerr1.cpu(),\
                                                              pred_y.cpu(), var.cpu()

    target_test_x = target_test_x.cpu()
        

    # Plot everything
    plt.plot(target_test_x[0], pred_y[0], 'b-', linewidth=1.5, label = 'mean model')
    plt.errorbar(target_x[0], target_y[0], yerr=yerr1[0], linestyle='', elinewidth=1.3, color='k',label = 'observations')
   # plt.plot(context_x[0], context_y[0], 'bo', markersize=1.5, label = 'context')
    
    plt.fill_between(
      target_test_x[0, :],
      pred_y[0, :] - var[0, :],
      pred_y[0, :] + var[0, :],
      alpha=0.6,
      facecolor='#ff9999', label='CI',
      interpolate=True)

    
    plt.legend(fontsize=10)

    # Make the plot pretty
    plt.yticks([-2, 0, 2])
    plt.xticks([-2, 0, 2])
    plt.ylim([-2, 2])
    plt.tick_params(axis='both', which='minor', labelsize=9)
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title(lcName, fontsize=8)
    
    if save:
        if isTrainData and flagval==0:
            savePath = OUTPUT_PATH + 'train/'
        elif notTraindata and flagval==1:
            savePath = OUTPUT_PATH + 'val/'
        else:
             savePath = OUTPUT_PATH + 'test/'
            
        lcName = lcName.split(',')[0]
        pltPath = os.path.join(savePath, 'plots', lcName + '.png')
        csvpath = os.path.join(savePath, 'data', lcName + '_predictions.csv')
        
        plt.savefig(pltPath, bbox_inches='tight')
        plt.clf()
        
        # Create dataframe with predictions and save csv
        d = {'time': target_test_x[0], 'cont': pred_y[0], 'conterr': var[0]}
        
        df = pd.DataFrame(data=d)
        
        df.to_csv(csvpath, index=False)
        
        if os.path.exists(pltPath + ".png") == False:
            print(pltPath)
    else:
        #plt.show()
        print('')

def load_test_data(DATA_PATH_TEST):
    testSet = LighCurvesDataset(root_dir = DATA_PATH_TEST, status = 'test')
    testLoader = DataLoader(testSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return testSet, testLoader

def load_train_data(DATA_PATH_TRAIN):
    trainSet = LighCurvesDataset(root_dir = DATA_PATH_TRAIN, status = 'test')
    trainLoader = DataLoader(trainSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return trainSet, trainLoader

def load_val_data(DATA_PATH_VAL):
    valSet = LighCurvesDataset(root_dir = DATA_PATH_VAL, status = 'test')
    valLoader = DataLoader(valSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return valSet, valLoader

def instantiate_pdf_document(testSet):
    plots = len(testSet)
    plots_per_page = 9 #num_rows X num_columns
    pages_count = int(np.ceil(plots / float(plots_per_page)))
    grid_size = (3, 3)
    count = 1
    
    # additional functionality can be added here
    
    return pages_count

def plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, device):
    pdf_directory = './output/predictions/test/plots'  # Hardcoded PDF directory path
    output_directory = './output/predictions/test'  # Hardcoded data directory path

    def generate_out_pdf_test(pdf_directory):
        # Create the directory if it doesn't exist
        os.makedirs(pdf_directory, exist_ok=True)

        # Set the PDF file path
        out_pdf_test = os.path.join(pdf_directory, 'test.pdf')

        return out_pdf_test

    grid_size = (3, 3)
    plots_per_page = 9
    pdf = matplotlib.backends.backend_pdf.PdfPages(generate_out_pdf_test(pdf_directory))
    i = 0
    j = 0
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
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Create a figure instance
            if i == 0 and j == 0:
                fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
                plt.tight_layout()

            if j == grid_size[1]:
                i += 1
                j = 0

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)

            plt.subplot2grid(grid_size, (i, j))
            plot_function(target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, save=False, isTrainData=True)

            j += 1

            # Saving data as a CSV file
            d = {'time': target_test_x[0], 
                 'cont': mu[0],  # Using mu for the predictions
                 'conterr': sigma[0]}  # Using sigma for the variance

            df = pd.DataFrame(data=d)

            # Specify the path for saving the CSV file
            csvpath = os.path.join(output_directory, 'data', lcName + '_predictions.csv')

            # Create the 'data' directory if it doesn't exist
            os.makedirs(os.path.dirname(csvpath), exist_ok=True)

            df.to_csv(csvpath, index=False)

            # Closing the page
            if ((i + 1) * (j + 1)) > plots_per_page:
                i = 0
                j = 0
                fig.text(0.5, 0.04, r'MJD', ha='center', va='center')
                fig.text(0.06, 0.5, r'Scaled magnitude', ha='center', va='center', rotation='vertical')
                pdf.savefig(fig)

        # Write the PDF document to the disk
        pdf.close()

    return testMetrics

def save_test_metrics(output_path, test_metrics):
    with open(output_path + 'test/testMetrics.json', 'w') as fp:
        json.dump(test_metrics, fp, indent=4)


def plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function, device):
    pdf_directory = './output/predictions/train/plots'  # Hardcoded PDF directory path
    output_directory = './output/predictions/train'  # Hardcoded data directory path

    def generate_out_pdf_train(pdf_directory):
        # Create the directory if it doesn't exist
        os.makedirs(pdf_directory, exist_ok=True)

        # Set the PDF file path
        out_pdf_train = os.path.join(pdf_directory, 'train.pdf')

        return out_pdf_train

    grid_size = (3, 3)
    plots_per_page = 9
    pdf = matplotlib.backends.backend_pdf.PdfPages(generate_out_pdf_train(pdf_directory))
    i = 0
    j = 0
    trainMetrics = {}

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

            # Add metrics to map
            trainMetrics[lcName] = {'log_prob:': str(loss),
                                   'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Create a figure instance
            if i == 0 and j == 0:
                fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
                plt.tight_layout()

            if j == grid_size[1]:
                i += 1
                j = 0

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)

            plt.subplot2grid(grid_size, (i, j))
            plot_function(target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, save=False, isTrainData=True)

            j += 1

            # Saving data as a CSV file
            d = {'time': target_test_x[0], 
                 'cont': mu[0],  # Using mu for the predictions
                 'conterr': sigma[0]}  # Using sigma for the variance

            df = pd.DataFrame(data=d)

            # Specify the path for saving the CSV file
            csvpath = os.path.join(output_directory, 'data', lcName + '_predictions.csv')

            # Create the 'data' directory if it doesn't exist
            os.makedirs(os.path.dirname(csvpath), exist_ok=True)

            df.to_csv(csvpath, index=False)

            # Closing the page
            if ((i + 1) * (j + 1)) > plots_per_page:
                i = 0
                j = 0
                fig.text(0.5, 0.04, r'MJD', ha='center', va='center')
                fig.text(0.06, 0.5, r'Scaled magnitude', ha='center', va='center', rotation='vertical')
                pdf.savefig(fig)

        # Write the PDF document to the disk
        pdf.close()

    return trainMetrics


def save_train_metrics(output_path, train_metrics):
    with open(output_path + 'train/trainMetrics.json', 'w') as fp:
        json.dump(train_metrics, fp, indent=4)

def plot_val_light_curves(model, valLoader, criterion, mseMetric, plot_function, device):
    pdf_directory = './output/predictions/val/plots'  # Hardcoded PDF directory path
    output_directory = './output/predictions/val'  # Hardcoded data directory path

    def generate_out_pdf_val(pdf_directory):
        # Create the directory if it doesn't exist
        os.makedirs(pdf_directory, exist_ok=True)

        # Set the PDF file path
        out_pdf_val = os.path.join(pdf_directory, 'val.pdf')

        return out_pdf_val

    grid_size = (3, 3)
    plots_per_page = 9
    pdf = matplotlib.backends.backend_pdf.PdfPages(generate_out_pdf_val(pdf_directory))
    i = 0
    j = 0
    valMetrics = {}

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

            # Add metrics to map
            valMetrics[lcName] = {'log_prob:': str(loss),
                                   'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))

            # Create a figure instance
            if i == 0 and j == 0:
                fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
                plt.tight_layout()

            if j == grid_size[1]:
                i += 1
                j = 0

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)

            plt.subplot2grid(grid_size, (i, j))
            plot_function(target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, save=False, isTrainData=True)

            j += 1

            # Saving data as a CSV file
            d = {'time': target_test_x[0], 
                 'cont': mu[0],  # Using mu for the predictions
                 'conterr': sigma[0]}  # Using sigma for the variance

            df = pd.DataFrame(data=d)

            # Specify the path for saving the CSV file
            csvpath = os.path.join(output_directory, 'data', lcName + '_predictions.csv')

            # Create the 'data' directory if it doesn't exist
            os.makedirs(os.path.dirname(csvpath), exist_ok=True)

            df.to_csv(csvpath, index=False)

            # Closing the page
            if ((i + 1) * (j + 1)) > plots_per_page:
                i = 0
                j = 0
                fig.text(0.5, 0.04, r'MJD', ha='center', va='center')
                fig.text(0.06, 0.5, r'Scaled magnitude', ha='center', va='center', rotation='vertical')
                pdf.savefig(fig)

        # Write the PDF document to the disk
        pdf.close()

    return valMetrics


def save_val_metrics(output_path, val_metrics):
    with open(output_path + 'val/valMetrics.json', 'w') as fp:
        json.dump(val_metrics, fp, indent=4)
