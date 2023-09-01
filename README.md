
# Instalation: 

pip install QNPy

# REQUIREMENTS

This package contain a requirements.txt file with all the requirements that needs to be setisfied (mainly other packages) before you can use it as single package. To install all requirements at once, you will need to:

1. in command line navigate to directory where you downloaded your package (where the requirements.txt file is)
2. once you are there type:
pip install -r requirements.txt

You are ready to use QNPy package now

# EXAMPLE: At this link: https://drive.google.com/file/d/1O7F_eCcGrWHyo3yxRYfFBGVr64rzFJBK/view?usp=sharing you can download a notebook that will take you through the entire process of using this package with an example of using each of the modules separately.

MODULES AND THEIR FUNCTIONS:

# 1. PREPROCESSING THE DATA

Preprocess.py

In this module we transform data in the range [-2,2]x[-2,2] to make training faster. It contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./preproc/ -- It is going to be the folder for saving the transformed data

Your data must contain: MJD or time, mag-magnitude and magerr-magnitude error 

__________________________________________________________
transform(data)
--Transforming data into [-2,2]x[-2,2] range. This function needs to be uploaded before using it
"""
    Args:
    data: Your data must contain: MJD or time, mag-magnitude and magerr-magnitude error

"""
_________________________________________________________
transform_and_save(files, DATA_SRC, DATA_DST, transform):

Transforms and saves a list of CSV files. Function also save tr coefficients as a pickle file named trcoeff.pickle
"""

    Args:
    files (list): A list of CSV or TXT file names.
    DATA_SRC (str): The path to the folder containing the CSV or TXT files.
    DATA_DST (str): The path to the folder where the transformed CSV or TXT files will be saved.
    transform: function for transformation defined previously

    Returns:
    list: A list of transformation coefficients for each file, where each element is a list containing the file name and the transformation coefficients Ax, Bx, Ay, and By.

    How to use:
    number_of_points, trcoeff = transform_and_save(files, DATA_SRC, DATA_DST, transform)

    After that you can plot number of data points
    """
___________________________________________________________

# 2. SPLITTING AND TRAINING THE DATA

SPLITTING_AND_TRAINING.py

We use this module to split the data into three subsamples that will serve as a test sample, a sample for model training, and a validation sample. It contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./dataset/train -- folder for saving data for traing after splitting your original dataset
2. ./dataset/test -- folder for test data 
3. ./dataset/val -- folder for validation data
4. ./output/cnp_model.pth -- folder where you are going to save your trained model

_________________________________________________________________

create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/'):

-- Creates a TRAIN, TEST and VAL folders in directory

"""
   how to use: create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/')

"""
___________________________________________________________
split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER)

-- Splits the data into TRAIN, TEST and VAL folders

"""
   You need to make sure that DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, and VAL_FOLDER are defined before calling the function split_data.

   Args:
       files (list): A list of CSV file names.
       DATA_SRC (string): Path to preproceed data
       TRAIN_FOLDER (string): Path for saving the train data
       TEST_FOLDER (string): Path for saving the test data
       VAL_FOLDER (string): Path for saving the validation data

   How to use: split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER)

"""
__________________________________________________________
# TRAINING 

Must contain:

DATA_PATH_TRAIN = "./dataset/train" # path to train folder
DATA_PATH_VAL = "./dataset/val" # path to val folder

MODEL_PATH = "./output/cnp_model.pth" # folder for saving model

Before running the training function you musti define:

BATCH_SIZE = 32# training batch size MUST REMAIN 32
EPOCHS = 6000 #This is optional
EARLY_STOPPING_LIMIT = 3000 #This is optional
________________________________________________________________________________
get_data_loader(data_path_train, data_path_val, batch_size)

--- Defining train and validation loader for training process and validation

"""
   Args:
   data_path_train(str): path to train folder
   data_path_val(str): path to val folder
   batch_size: it is recommended to be 32

   How to use: trainLoader, valLoader = get_data_loader(DATA_PATH_TRAIN,BATCH SIZE)
"""

__________________________________________________________
create_model_and_optimizer()

--Defines the model as Deterministic Model, optimizer as torch optimizer, criterion as LogProbLoss, mseMetric as MSELoss and maeMetric as MAELoss
"""
   How to use: model, optimizer, criterion, mseMetric, maeMetric = create_model_and_optimizer(device)
   Device has to be defined before and it can be cuda or cpu
"""
__________________________________________________________
train_model(model, train_loader, val_loader,criterion, optimizer, num_runs, epochs, early_stopping_limit, mse_metric, maeMetric, device)

-- Trains the model

"""
   Args:
       model: Deterministic model
       train_loader: train loader
       val_loader: validation loader
       criterion: criterion
       optimizer: torch optimizer
       num_runs: The number of trainings 
       epochs: epochs for training. This is optional, but minimum of 3000 is recomended
       early_stopping_limit: limits the epochs for stopping the training. This is optional but minimum of 1500 is recomended
       mse_metric: mse metric
       mae_metric: mae metric
       device: torch device cpu or cuda

   How to use: If you want to save history_loss_train, history_loss_val, history_mse_train and history_mse_val for plotting you train your model like:

history_loss_train, history_loss_val, history_mse_train, history_mse_val, history_mae_train, history_mae_val, epoch_counter_train_loss, epoch_counter_train_mse, epoch_counter_train_mae, epoch_counter_val_loss, epoch_counter_val_mse, epoch_counter_val_mae = st.train_model(model, trainLoader, valLoader, criterion, optimizer, 1, 3000, 1500, mseMetric, maeMetric, device)

"""
__________________________________________________________
save_lists_to_csv(file_names,lists)

--saving the histories to lists

"""
   args:
   file_names: A list of file names to be used for saving the data. Each file name corresponds to a specific data list that will be saved in CSV format.
   lists (list): A list of lists containing the data to be saved. Each inner list represents a set of rows to be written to a CSV file.

   How to use: 
   # Define the file names for saving the lists
file_names = ["history_loss_train.csv", "history_loss_val.csv", "history_mse_train.csv", "history_mse_val.csv","history_mae_train.csv", "history_mae_val.csv", "epoch_counter_train_loss.csv", "epoch_counter_train_mse.csv", "epoch_counter_train_mae.csv", "epoch_counter_val_loss.csv","epoch_counter_val_mse.csv", "epoch_counter_val_mae.csv"]

# Define the lists
lists = [history_loss_train, history_loss_val, history_mse_train, history_mse_val, history_mae_train,
         history_mae_val, epoch_counter_train_loss, epoch_counter_train_mse, epoch_counter_train_mae,
         epoch_counter_val_loss, epoch_counter_val_mse, epoch_counter_val_mae]

save_list= save_lists_to_csv(file_names, lists)
__________________________________________________________
plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file)

-- plotting the history losses

"""
   args:
   returned data from test_model
   How to use: 
   
   history_loss_train_file = './history_loss_train.csv'  # Replace with the path to your history_loss_train CSV file
history_loss_val_file = './history_loss_val.csv'  # Replace with the path to your history_loss_val CSV file
epoch_counter_train_loss_file = './epoch_counter_train_loss.csv'  # Replace with the path to your epoch_counter_train_loss CSV file
   
   logprobloss=plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file)

"""
__________________________________________________________
plot_mse_metric(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file)

-- plotting the mse metric

"""
   args:
   returned data from test_model
   How to use: 
   
   history_mse_train_file = './history_mse_train.csv'  # Replace with the path to your history_mse_train CSV file
history_mse_val_file = './history_mse_val.csv'  # Replace with the path to your history_mse_val CSV file
epoch_counter_train_mse_file = './epoch_counter_train_mse.csv'  # Replace with the path to your epoch_counter_train_mse CSV file
   
   msemetric=plot_mse(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file)

"""

__________________________________________________________

plot_mae_metric(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file)

-- plotting the mae metric

"""
   args:
   returned data from test_model
   How to use: 
   
   history_mae_train_file = './history_mae_train.csv'  # Replace with the path to your history_mae_train CSV file
history_mae_val_file = './history_mae_val.csv'  # Replace with the path to your history_mae_val CSV file
epoch_counter_train_mae_file = './epoch_counter_train_mae.csv'  # Replace with the path to your epoch_counter_train_mae CSV file
   
   maemetric=plot_mae(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file)
"""
__________________________________________________________
save_model(model, MODEL_PATH)

-- saving the model

"""
   Args:
   model: Deterministic model
   MODEL_PATH(str): output path for saving the model

   How to use: save_model(model, MODEL_PATH)
"""
__________________________________________________________
# 3. PREDICTION AND PLOTTING THE TRANSFORMED DATA, EACH CURVE INDIVIDUALLY

PREDICTION.py

We use this module for prediction and plotting of models of transformed data. Each curve will be plotted separately. It contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./output/predictions/train/plots -- folder for saving training plots
2. ./output/predictions/test/plots -- folder for saving test plots 
3. ./output/predictions/val/plots -- folder for saving validation plots
4. ./output/predictions/train/data -- folder for sving train data
5. ./output/predictions/test/data -- folder for saving test data
6. ./output/predictions/val/data -- folder for saving val data
_____________________________________________________________
prepare_output_dir(OUTPUT_PATH)

-- the function prepare_output_dir takes the OUTPUT_PATH as an argument and removes all files in the output directory using os.walk method.

"""
   Args:
      OUTPUT_PATH(str): path to output folder

   How to use: prepare_output_dir(OUTPUT_PATH)
"""
__________________________________________________________
load_trained_model(MODEL_PATH, device)

--Uploading trained model

"""
   agrs:
   MODEL_PATH(str) = path to model directorium
   device = torch device CPU or CUDA
   How to use: model=load_trained_model(MODEL_PATH, device)
"""
__________________________________________________________
get_criteria()

-- Gives the criterion and mse_metric

"""
   How to use: criterion, mseMetric=get_criteria()
"""
__________________________________________________________
plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save = False, flagval=0, isTrainData = None, notTrainData = None):

-- Defines the plots of the light curve data and predicted mean and variance, and it should be imported separately
    """

    Args:
    context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    """
__________________________________________________________
load_test_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to Test data

   How to use: testLoader=load_test_data(DATA_PATH_TEST)

"""
__________________________________________________________
load_train_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to train data

   How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

"""
_________________________________________________________
load_val_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to VAL data

   How to use: valLoader=load_val_data(DATA_PATH_VAL)

"""
_________________________________________________________

plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device)

-- Ploting the transformed light curves from test set

"""
   Args:
       model: Deterministic model
       testLoader: Uploaded test data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA

       How to use: testMetrics = plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device)
"""
_________________________________________________________

save_test_metrics(OUTPUT_PATH, testMetrics)

-- saving the test metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       testMetrics: returned data from ploting function

       How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
"""
_________________________________________________________

plot_light_curves_from_train_set(model, trainLoader, criterion, mseMetric, plot_function, device)

-- Ploting the transformed light curves from train set

"""
   Args:
       model: Deterministic model
       trainLoader: Uploaded trained data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA

       How to use: trainMetrics = plot_light_curves_from_train_set(model, trainLoader, criterion, mseMetric, plot_function, device)
"""
_________________________________________________________

save_train_metrics(OUTPUT_PATH, testMetrics)

-- saving the train metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       trainMetrics: returned data from ploting function

       How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
"""
_________________________________________________________

plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device)

-- Ploting the transformed light curves from validation set

"""
   Args:
       model: Deterministic model
       valLoader: Uploaded val data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA

       How to use: valMetrics = plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device)
"""
_________________________________________________________
save_val_metrics(OUTPUT_PATH, valMetrics)

-- saving the validation metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       valMetrics: returned data from ploting function

       How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
"""
_________________________________________________________

# 4. PREDICTION AND PLOTTING THE TRANSFORMED DATA, IN ONE PDF FILE

PREDICTION_onePDF.py

We use this module for prediction and plotting of models of transformed data. All curves will be plotted in one PDF file. This module contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./output/predictions/train/plots -- folder for saving training plots
2. ./output/predictions/test/plots -- folder for saving test plots 
3. ./output/predictions/val/plots -- folder for saving validation plots
4. ./output/predictions/train/data -- folder for sving train data
5. ./output/predictions/test/data -- folder for saving test data
6. ./output/predictions/val/data -- folder for saving val data

_________________________________________________________
clear_output_dir(output_path)

-- Removes all files in the specified output directory.
    """

    Args:
        output_path (str): The path to the output directory.

        How to use: clear_output_dir(OUTPUT_PATH)

    """
_________________________________________________________
load_model(model_path, device):
--Loads a trained model from disk and moves it to the specified device.
    """
    Args:
        model_path (str): The path to the saved model.
        device (str or torch.device): The device to load the model onto, CPU or CUDA

        How to use: model = load_model(MODEL_PATH, device)
    """
_________________________________________________________
get_criteria()

-- Gives the criterion and mse_metric

"""
   How to use: criterion, mseMetric=get_criteria()
"""
_________________________________________________________
plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save = False, flagval=0, isTrainData = None, notTrainData = None):

-- Defines the plots of the light curve data and predicted mean and variance
    """

    Args:
    context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    """
_________________________________________________________
load_test_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to Test data

   How to use: testLoader=load_test_data(DATA_PATH_TEST)

"""
__________________________________________________________
load_train_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to train data

   How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

"""
_________________________________________________________
load_val_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to VAL data

   How to use: valLoader=load_val_data(DATA_PATH_VAL)

"""
_________________________________________________________
instantiate_pdf_document(testSet)

-- Formating the PDF pages
   Default: plots per page = 9

   """
      Args:
      testSet: from testLoader

      how to use: out_pdf_test=instantiate_pdf_document(testSet)
                  out_pdf_train=instantiate_pdf_document(trainSet)
                  out_pdf_val=instantiate_pdf_document(valSet)
   """
_________________________________________________________
plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, out_pdf, device)
-- ploting the test set in range [-2,2]

"""
   Args:
   model: model
   testLoader: Test set
   criterion: criterion
   mseMetric: mse Metric
   plot_function: defined above
   out_pdf: pdf inisiated before
   device: Torch device, CPU or CUDA


   how to use: testMetrics=plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, out_pdf_test, device)
"""
_________________________________________________________
save_test_metrics(OUTPUT_PATH, testMetrics)

-- saving the test metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       testMetrics: returned data from ploting function

       How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
"""
_________________________________________________________

plot_train_set(model, trainLoader, device, criterion, mseMetric, plot_function, out_pdf)

-- Ploting the transformed light curves from train set

"""
   Args:
       model: Deterministic model
       trainLoader: Uploaded trained data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       out_pdf: pdf inisiated before

       How to use: trainMetrics=plot_train_set(model, trainLoader, device, criterion, mseMetric, plot_function, out_pdf_train)
"""
_________________________________________________________

save_train_metrics(OUTPUT_PATH, testMetrics)

-- saving the train metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       trainMetrics: returned data from ploting function

       How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
"""
_________________________________________________________

create_val_plots(model, valLoader, device, criterion, mseMetric, plot_function, out_pdf)

-- Ploting the transformed light curves from val set

"""
   Args:
       model: Deterministic model
       valLoader: Uploaded val data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       out_pdf: pdf inisiated before

       How to use: valMetrics = create_val_plots(model, valLoader, device, criterion, mseMetric, plot_function, out_pdf_val)
"""
_________________________________________________________
save_val_metrics(OUTPUT_PATH, valMetrics)

-- saving the val metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       valMetrics: returned data from ploting function

       How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
"""
_________________________________________________________

# 5. PREDICTION AND PLOTTING THE DATA IN ORIGINAL DATA RANGE, EACH CURVE INDIVIDUALLY

PREDICTION_Original_mjd.py

We use this module to predict and plot the model in the original range of data. All curves are plotted individually. This module contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./output/predictions/train/plots -- folder for saving training plots
2. ./output/predictions/test/plots -- folder for saving test plots 
3. ./output/predictions/val/plots -- folder for saving validation plots
4. ./output/predictions/train/data -- folder for sving train data
5. ./output/predictions/test/data -- folder for saving test data
6. ./output/predictions/val/data -- folder for saving val data

__________________________________________________________________

prepare_output_dir(OUTPUT_PATH)

-- the function prepare_output_dir takes the OUTPUT_PATH       as an argument and removes all files in the output    directory using os.walk method.

"""
   Args:
      OUTPUT_PATH(str): path to output folder

   How to use: prepare_output_dir(OUTPUT_PATH)
"""
__________________________________________________________
load_trained_model(MODEL_PATH, device)

--Uploading trained model

"""
   agrs:
   MODEL_PATH(str) = path to model directorium
   device = torch device CPU or CUDA
   How to use: model=load_trained_model(MODEL_PATH, device)
"""
__________________________________________________________
get_criteria()

-- Gives the criterion and mse_metric

"""
   How to use: criterion, mseMetric=get_criteria()
"""
__________________________________________________________
load_trcoeff()

-- loading the original coefficients from pickle file

"""
   How to use: tr=load_trcoeff()

"""
_________________________________________________________
plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, isTrainData = None, flagval = 0, notTrainData = None)

-- function for ploting the light curves

"""
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
_________________________________________________________
load_test_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to Test data

   How to use: testLoader=load_test_data(DATA_PATH_TEST)

"""
__________________________________________________________
load_train_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to train data

   How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

"""
_________________________________________________________
load_val_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to VAL data

   How to use: valLoader=load_val_data(DATA_PATH_VAL)

"""
_________________________________________________________
plot_test_data(model, testLoader, criterion, mseMetric, plot_function, device, tr)

-- Ploting the light curves from test set in original mjd range

"""
   Args:
       model: Deterministic model
       testLoader: Uploaded test data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       tr: trcoeff from pickle file

       How to use: testMetrics=plot_test_data(model, testLoader, criterion, mseMetric, plot_function2, device, tr)
"""
_________________________________________________________

save_test_metrics(OUTPUT_PATH, testMetrics)

-- saving the test metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       testMetrics: returned data from ploting function

       How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
"""
_________________________________________________________
plot_train_light_curves(trainLoader, model, criterion, mseMetric, plot_function, device, tr)

-- Ploting the light curves from train set in original mjd range

"""
   Args:
       model: Deterministic model
       trainLoader: Uploaded trained data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       tr: trcoeff from pickle file

       How to use: trainMetrics=plot_train_light_curves(trainLoader, model, criterion, mseMetric, plot_function2, device,tr)
"""
_________________________________________________________

save_train_metrics(OUTPUT_PATH, testMetrics)

-- saving the train metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       trainMetrics: returned data from ploting function

       How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
"""
_________________________________________________________
plot_val_curves(model, valLoader, criterion, mseMetric, plot_function, device, tr)

-- Ploting the light curves from val set in original mjd range

"""
   Args:
       model: Deterministic model
       valLoader: Uploaded val data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       tr: trcoeff from pikle file

       How to use: valMetrics=plot_val_curves(model, valLoader, criterion, mseMetric, plot_function2, device,tr)
"""
_________________________________________________________
save_val_metrics(OUTPUT_PATH, valMetrics)

-- saving the val metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       valMetrics: returned data from ploting function

       How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
"""
_________________________________________________________

# 6. PREDICTION AND PLOTTING THE DATA IN ORIGINAL DATA RANGE, IN ONE PDF FILE

PREDICTION_onePDF_original_mjd.py

We use this module to predict and plot the model in the original range of data. All curves are plotted in one PDF file. This module contains the following functions that must be executed in order.

Before running this script, you must create the following folders in the directory where your Python notebook is located:
1. ./output/predictions/train/plots -- folder for saving training plots
2. ./output/predictions/test/plots -- folder for saving test plots 
3. ./output/predictions/val/plots -- folder for saving validation plots
4. ./output/predictions/train/data -- folder for sving train data
5. ./output/predictions/test/data -- folder for saving test data
6. ./output/predictions/val/data -- folder for saving val data

____________________________________________________________

clear_output_dir(output_path)

-- Removes all files in the specified output directory.
    """

    Args:
        output_path (str): The path to the output directory.

        How to use: clear_output_dir(OUTPUT_PATH)

    """
_________________________________________________________
load_model(model_path, device):
--Loads a trained model from disk and moves it to the specified device.
    """
    Args:
        model_path (str): The path to the saved model.
        device (str or torch.device): The device to load the model onto, CPU or CUDA

        How to use: model = load_model(MODEL_PATH, device)
    """
_________________________________________________________
get_criteria()

-- Gives the criterion and mse_metric

"""
   How to use: criterion, mseMetric=get_criteria()
"""
_________________________________________________________
load_trcoeff()

-- loading the original coefficients from pickle file

"""
   How to use: tr=load_trcoeff()

"""
_________________________________________________________
plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, save = False, isTrainData = None, flagval = 0, notTrainData = None)

-- function for ploting the light curves. It needs to be uploaded separately

"""
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
_________________________________________________________
load_test_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to Test data

   How to use: testLoader=load_test_data(DATA_PATH_TEST)

"""
__________________________________________________________
load_train_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to train data

   How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)

"""
_________________________________________________________
load_val_data(data_path)

-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches
"""
   Args:
       data_path(str): path to VAL data

   How to use: valLoader=load_val_data(DATA_PATH_VAL)

"""
_________________________________________________________
instantiate_pdf_document(testSet)

-- Formating the PDF pages
   Default: plots per page = 9

   """
      Args:
      testSet: from testLoader

      how to use: out_pdf_test=instantiate_pdf_document(testSet)
      out_pdf_train=instantiate_pdf_document(trainSet)
      out_pdf_val=instantiate_pdf_document(valSet)
   """
_________________________________________________________

plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function, out_pdf, device,tr)

-- ploting the test set in original range

"""
   Args:
   model: model
   testLoader: Test set
   criterion: criterion
   mseMetric: mse Metric
   plot_function: defined above
   out_pdf: pdf inisiated before
   device: Torch device, CPU or CUDA
   tr: trcoeff from pickle file


   how to use: testMetrics=plot_test_light_curves(model, testLoader, criterion, mseMetric, plot_function2, out_pdf_test, device,tr)
"""
_________________________________________________________
save_test_metrics(OUTPUT_PATH, testMetrics)

-- saving the test metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       testMetrics: returned data from ploting function

       How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
"""
_________________________________________________________
plot_train_set(model, trainLoader, device, criterion, mseMetric, plot_function, out_pdf,tr)

-- Ploting the light curves from train set in original mjd range

"""
   Args:
       model: Deterministic model
       trainLoader: Uploaded trained data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       out_pdf: pdf inisiated before
       tr: trcoeff from pickle file

       How to use: trainMetrics=plot_train_set(model, trainLoader, device, criterion, mseMetric, plot_function2, out_pdf_train,tr)
"""
_________________________________________________________

save_train_metrics(OUTPUT_PATH, testMetrics)

-- saving the train metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       trainMetrics: returned data from ploting function

       How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
"""
_________________________________________________________
create_val_plots(model, valLoader, device, criterion, mseMetric, plot_function, out_pdf,tr)

-- Ploting the light curves from val set in original mjd range

"""
   Args:
       model: Deterministic model
       valLoader: Uploaded val data
       criterion: criterion
       mseMetric: Mse Metric
       plot_function: plot function defined above
       device: torch device CPU or CUDA
       out_pdf: pdf inisiated before
       tr: trcoeff from pickle file

       How to use: valMetrics=create_val_plots(model, valLoader, device, criterion, mseMetric, plot_function2, out_pdf_val,tr)
"""
_________________________________________________________
save_val_metrics(OUTPUT_PATH, valMetrics)

-- saving the val metrics as json file

"""
   Args:
       OUTPUT_PATH(str): path to output folder
       valMetrics: returned data from ploting function

       How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
"""
_________________________________________________________

# THE END

