import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
import os
import glob 
import warnings
import dill

def transform(data):
    x_data = np.array(data['mjd'])
    y_data = np.array(data['mag'])
    z_data = np.array(data['magerr'])
    
    # Skip LCs with low number of points
    if len(x_data) < 100:
        return None
    
    # If magerr is 0 replace it with 0.01*mag
    for i in range(len(z_data)):
        if z_data[i] == 0:
            z_data[i] = 0.01 * y_data[i]
    
    Mmean=np.mean(y_data)
    #normalization with measurment errors ###
    #y_data = (y_data - np.mean(y_data))/(z_data)
    
   
    ### mapping time and flux to interval [-2,2]
    a = -2
    b = 2

    Ax = min(x_data)
    Bx = max(x_data)

    Ay = min(y_data)
    By = max(y_data)
   
    #make series with added noise
    y_dataeplus=y_data+z_data
    #make series with subtracted noise
    y_dataeminus=y_data-z_data

    Ayeplus = min(y_dataeplus)
    Byeplus = max(y_dataeplus)

    Ayeminus = min(y_dataeminus)
    Byeminus= max(y_dataeminus)

    x_data = (x_data - Ax)*(b-a)/(Bx - Ax) + a
    y_data = (y_data - Ay)*(b-a)/(By - Ay) + a
    
     
    y_dataeplus1 = (y_dataeplus - Ayeplus)*(b-a)/(Byeplus - Ayeplus) + a
    y_dataeminus1 = (y_dataeminus - Ayeminus)*(b-a)/(Byeminus - Ayeminus) + a




   # z_data1=(z_data/np.abs(y_data))*np.abs(y_data)
    
    data = {'time':    x_data,
            'cont':    y_data,
            'conterr': z_data}
    data = pd.DataFrame(data)
    dataeplus = {'time':    x_data,
            'cont':    y_dataeplus1,
            'conterr': z_data}
    dataeplus = pd.DataFrame(dataeplus)
    dataeminus = {'time':    x_data,
            'cont':    y_dataeminus,
            'conterr': z_data}
    dataeminus = pd.DataFrame(dataeminus)


    return data,dataeplus,dataeminus,Ax,Bx, Ay,By

def transform_and_save(files, data_src, data_dst, transform):
    number_of_points = []
    counter = 0
    trcoeff = []
    for file in files:
        lcName = file.split(".")[0]
        tmpDataFrame = pd.read_csv(os.path.join(data_src, file))
        # Catch runtime warnings in transform function
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                data_transformed, _, _, Ax, Bx, Ay, By = transform(tmpDataFrame)
            except Warning as e:
                print('warning found:', e)
                print(lcName)
                continue
            except Exception as e:
                print('error found:', e)
                print(lcName)
                continue
        if data_transformed is None:
            continue
        counter += 1
        number_of_points.append(len(data_transformed.index))
        filename = os.path.join(data_dst, file)
        data_transformed.to_csv(filename, index=False)
        trcoeff.append([lcName, Ax, Bx, Ay, By])
        
        dill.dump(trcoeff, file = open("trcoeff.pickle", "wb"))
    return number_of_points, trcoeff


