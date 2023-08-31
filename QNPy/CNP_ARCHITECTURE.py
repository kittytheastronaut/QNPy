import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

class DeterministicEncoder(nn.Module):
    """The Encoder."""

    def __init__(self):
        """CNP encoder."""
        super(DeterministicEncoder, self).__init__()

        self.linearInput = nn.Linear(  2, 128)
        self.linear1     = nn.Linear(128, 128)
        self.linear2     = nn.Linear(128, 128)
        self.linear3     = nn.Linear(128, 128)
        self.relu        = nn.ReLU()

    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation.

        Args:
          context_x, context_y: Tensors of size BATCH_SIZE x NUM_OF_OBSERVATIONS

        Returns:
          representation: The encoded representation averaged over all context points.
        """

        # Concatenate x and y along the filter axes
        context_x = torch.unsqueeze(context_x, -1)
        context_y = torch.unsqueeze(context_y, -1)
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        # Reshape to parallelise across observations
        batch_size, num_context_points, filter_size = encoder_input.shape
        input = torch.reshape(encoder_input, (batch_size * num_context_points, -1))

        # Pass through layers
        x = self.relu(self.linearInput(input))
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        # Bring back into original shape
        x = torch.reshape(x, (batch_size, num_context_points, 128))

        # Aggregator: take the mean over all points
        out = torch.mean(x, dim=1)

        return out
    
class DeterministicDecoder(nn.Module):
    """The Decoder."""

    def __init__(self):
        """CNP decoder."""
        super(DeterministicDecoder, self).__init__()

        self.linearInput  = nn.Linear(129, 128)
        self.linear1      = nn.Linear(128, 128)
        self.linear2      = nn.Linear(128, 128)
        self.linearOutput = nn.Linear(128, 2)
        self.relu         = nn.ReLU()
        self.softplus     = nn.Softplus()

    def forward(self, representation, target_x):
        """Decodes the individual targets.

        Args:
          representation: The encoded representation of the context
          target_x: The x locations for the target query

        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
        """
        num_total_points = target_x.shape[1]

        # Concatenate the representation and the target_x
        representation = torch.tile(
            torch.unsqueeze(representation, dim=1), 
            [1, num_total_points, 1]
        )
        target_x = torch.unsqueeze(target_x, -1)

        decoder_input = torch.cat((representation, target_x), dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = decoder_input.shape
        input = torch.reshape(decoder_input, (batch_size * num_total_points, -1))

        # Pass through layers
        x = self.relu(self.linearInput(input))
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linearOutput(x)

        # Bring back into original shape
        out = torch.reshape(x, (batch_size, num_total_points, -1))

        # Get the mean an the variance
        mu, log_sigma = torch.split(out, 1, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * self.softplus(log_sigma)

        # Squeeze last dim
        mu = torch.squeeze(mu, dim=-1)
        sigma = torch.squeeze(sigma, dim=-1)

        # Get cov matrix
        cov = torch.stack([torch.diag(x) for x in torch.unbind(sigma)])              

        # Get the distribution
        dist = MultivariateNormal(mu, cov)

        return dist, mu, sigma
    
class DeterministicModel(nn.Module):
    """The CNP model."""

    def __init__(self):
        """Initializes the model."""
        super(DeterministicModel, self).__init__()

        self._encoder = DeterministicEncoder()
        self._decoder = DeterministicDecoder()

    def forward(self, context_x, context_y, target_x):
        """Returns the predicted mean and variance at the target points.

        Args:
          context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
          context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
          target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
          
        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
        """

        representation = self._encoder(context_x, context_y)
        dist, mu, sigma =  self._decoder(representation, target_x)

        return dist, mu, sigma