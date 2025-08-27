# PyTorch Elements in delta2.py

This document lists PyTorch elements found in the `delta2.py` file, categorized by their primary function or "goal" within the context of the delta prediction task.

## Goal 1: Data Preparation and Augmentation

These elements are primarily used for handling, processing, and augmenting data for model training.

*   **Classes:**
    *   <mcsymbol name="EnhancedDeltaDataset" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="23" type="class"></mcsymbol>: Custom PyTorch `Dataset` for handling features and targets, with optional data augmentation.
*   **Modules/Functions:**
    *   `torch.utils.data.Dataset`: Base class for custom datasets.
    *   `torch.utils.data.DataLoader`: Utility for loading data in batches.
    *   `torch.from_numpy`: Converts NumPy arrays to PyTorch tensors.
    *   `torch.randperm`: Generates a random permutation of integers, useful for splitting data.
    *   `torch.normal`: Generates random numbers from a normal distribution, used for adding noise during augmentation.

## Goal 2: Model Architectures

This section includes the various neural network models and their building blocks.

*   **Classes:**
    *   `torch.nn as nn`: Base module for all neural network layers.
    *   <mcsymbol name="TransformerDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="39" type="class"></mcsymbol>: A Transformer-based model for delta prediction.
        *   `nn.Linear`: Applies a linear transformation.
        *   `nn.Parameter`: Tensors that are considered a module parameter.
        *   `nn.TransformerEncoderLayer`: A single layer of a Transformer encoder.
        *   `nn.TransformerEncoder`: A stack of Transformer encoder layers.
        *   `nn.MultiheadAttention`: Implements multi-head attention mechanism.
        *   `nn.Sequential`: A sequential container of modules.
        *   `nn.GELU`: Gaussian Error Linear Unit activation function.
        *   `nn.Dropout`: Randomly zeroes some of the elements for regularization.
        *   `nn.Sigmoid`: Sigmoid activation function.
    *   <mcsymbol name="CNNFeatureExtractor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="106" type="class"></mcsymbol>: A 1D Convolutional Neural Network for feature extraction.
        *   `nn.Conv1d`: Applies a 1D convolution over an input signal.
        *   `nn.BatchNorm1d`: Applies Batch Normalization over a 2D or 3D input.
        *   `nn.AdaptiveAvgPool1d`: Applies a 1D adaptive average pooling.
    *   <mcsymbol name="LSTMDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="135" type="class"></mcsymbol>: An LSTM-based model for sequential data.
        *   `nn.LSTM`: Long Short-Term Memory layer.
        *   `nn.Tanh`: Hyperbolic tangent activation function.
    *   <mcsymbol name="ResidualBlock" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="169" type="class"></mcsymbol>: A basic residual block for deep networks.
    *   <mcsymbol name="EnsembleDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="190" type="class"></mcsymbol>: An ensemble model combining multiple architectures (<mcsymbol name="TransformerDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="39" type="class"></mcsymbol>, <mcsymbol name="LSTMDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="135" type="class"></mcsymbol>, <mcsymbol name="CNNFeatureExtractor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="106" type="class"></mcsymbol>).
        *   `nn.ModuleList`: Holds submodules in a list.
        *   `nn.ReLU`: Rectified Linear Unit activation function.
    *   `nn.Softplus`: Softplus activation function, used in the uncertainty model.
*   **Functions:**
    *   `torch.nn.functional as F`: Contains functional implementations of neural network operations (e.g., `F.relu`, `F.softmax`).

## Goal 3: Training and Optimization

These elements are crucial for defining the training process, including loss calculation, parameter updates, and learning rate adjustments.

*   **Modules/Functions:**
    *   `torch.optim as optim`: Package implementing various optimization algorithms.
    *   `optim.AdamW`: AdamW optimizer.
    *   `optim.RMSprop`: RMSprop optimizer.
    *   `torch.optim.lr_scheduler.ReduceLROnPlateau`: (Imported but not explicitly used in the provided code snippet) Adjusts learning rate when a metric has stopped improving.
    *   `torch.optim.lr_scheduler.CosineAnnealingLR`: Sets the learning rate of each parameter group using a cosine annealing schedule.
    *   `torch.optim.lr_scheduler.OneCycleLR`: Sets the learning rate according to the 1cycle policy.
    *   `nn.MSELoss`: Mean Squared Error loss function.
    *   `nn.HuberLoss`: Huber loss function, less sensitive to outliers than MSE.
    *   `torch.nn.utils.clip_grad_norm_`: Clips gradients to prevent exploding gradients.
    *   `torch.nn.init.xavier_uniform_`: Initializes weights using a Xavier uniform distribution.
    *   `torch.nn.init.zeros_`: Initializes biases to zeros.
    *   `torch.cuda.amp.GradScaler`: Helps perform mixed precision training.
    *   `torch.cuda.amp.autocast`: Context manager for mixed precision training.
    *   `torch.compile`: Optimizes models for faster execution (if available).

## Goal 4: Prediction and Evaluation

Elements used for generating predictions from trained models and potentially evaluating their output.

*   **Modules/Functions:**
    *   `torch.no_grad()`: Context-manager that disables gradient calculation, useful for inference.
    *   `torch.stack`: Concatenates sequence of tensors along a new dimension.
    *   `torch.sum`: Calculates the sum of all elements in the input tensor.
    *   `F.softmax`: Applies the softmax function, used for normalizing ensemble weights.

## Goal 5: Utility and Infrastructure

General PyTorch utilities, device management, and overarching control structures.

*   **Modules/Functions:**
    *   `torch`: The main PyTorch library, providing core tensor operations.
    *   `torch.device`: Represents the device on which a `torch.Tensor` is or will be allocated.
    *   `torch.cuda.is_available()`: Checks if CUDA is available.
    *   `torch.cuda.get_device_name()`: Returns the name of the current CUDA device.
    *   `torch.ones`: Creates a tensor filled with the scalar value 1.
    *   `torch.cat`: Concatenates the given sequence of tensors in the given dimension.
    *   <mcsymbol name="EnhancedLiveDeltaPredictor" filename="delta2.py" path="/Users/adamaslan/code/ai-fin-opt2/ai-fin3/delta2.py" startline="279" type="class"></mcsymbol>: The main class orchestrating the entire process, including model initialization, training, and prediction, and managing device placement.