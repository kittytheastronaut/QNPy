import random
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch

class LighCurvesDataset(Dataset):
    """Dataset class."""
    
    def __init__(self, root_dir, status):
        self.root_dir = root_dir
        self.status = status
        self.file_paths = []
        for file_name in os.listdir(self.root_dir):
            self.file_paths.append(os.path.join(self.root_dir, file_name))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Reads one light curve data and picks target and context points."""
        
        # Read data
        lcName = os.path.basename(self.file_paths[idx])
        data = pd.read_csv(self.file_paths[idx])
        x_data = data['time']
        y_data = data['cont'] 
        z_data = data['conterr']  ### input error

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        num_total = np.size(x_data)
        
        context_x = np.copy(x_data)
        context_y = np.copy(y_data)

        target_x = np.copy(x_data)
        target_y = np.copy(y_data)
        
        measurement_error = np.copy(z_data)
        
        if self.status == 'test':
            # Select targets on the whole curve
            target_test_x = np.linspace(-2., 2., num = 400)

            
        # Convert to tensors
        target_x = torch.Tensor(target_x)
        target_y = torch.Tensor(target_y)
        context_x = torch.Tensor(context_x)
        context_y = torch.Tensor(context_y)
        measurement_error = torch.Tensor(measurement_error)
        # Squeeze first dimension
        target_x = torch.squeeze(target_x, 0)
        target_y = torch.squeeze(target_y, 0)
        context_x = torch.squeeze(context_x, 0)
        context_y = torch.squeeze(context_y, 0)
        measurement_error = torch.squeeze(measurement_error, 0)
      
        if self.status == 'train': 
            data = {'lcName' : lcName,
                    'context_x': context_x,
                    'context_y': context_y,
                    'target_x': target_x,
                    'target_y': target_y,
                    'measurement_error': measurement_error}
        else:
            target_test_x = torch.Tensor(target_test_x)
            target_test_x = torch.squeeze(target_test_x)

            data = {'lcName' : lcName,
                    'context_x': context_x,
                    'context_y': context_y,
                    'target_x': target_x,
                    'target_y': target_y,
                    'measurement_error': measurement_error,
                    'target_test_x': target_test_x}

        return data
    
def collate_lcs(batch):
    """Custom collate function for padding and stacking tensors in a batch.
    
    Args:
          batch: List containing variable length tensors where each item represents data for one light curve
                     data = {'lcName' : lcName,
                             'context_x': context_x,
                             'context_y': context_y,
                             'target_x': target_x,
                             'target_y': target_y,
                             'measurement_error': z_data}
        Returns:
          [context_x, context_y, target_x], target_y: Padded and stacked tensors.
    """
    
    # Calculate max num_total points
    num_total = None
    for item in batch:
        if num_total is None:
            num_total = item['context_x'].shape[0]
        else:
            num_total = min(num_total, item['context_x'].shape[0])
        
    # Determine number of context points
    upper_bound = int(num_total * 80/100)      # 80% of total points
    lower_bound = int(num_total * 60/100)      # 60% of total points
    num_context = random.randint(lower_bound, upper_bound)
    
    # Determine number of target points
    num_target = random.randint(num_context, num_total)
    
    context_x = []
    context_y = []
    target_x  = []
    target_y  = []
    measurement_error = []
    for item in batch:
        # Pad and append to list
        context_x.append(item['context_x'][:num_context]) 
        context_y.append(item['context_y'][:num_context]) 
        target_x.append(item['target_x'][:num_target]) 
        target_y.append(item['target_y'][:num_target]) 
        measurement_error.append(item['measurement_error'][:num_target]) 
    
    # Stack tensors
    context_x = torch.stack(context_x)
    context_y = torch.stack(context_y)
    target_x  = torch.stack(target_x)
    target_y  = torch.stack(target_y)
    measurement_error = torch.stack(measurement_error)
    
    return [context_x, context_y, target_x, measurement_error], target_y