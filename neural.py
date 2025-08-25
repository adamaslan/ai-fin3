# analyze_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Suppress Matplotlib warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## -----------------------------------------------------------
## SECTION 0: DUMMY MODEL AND DATA SETUP
## -----------------------------------------------------------
# This section is for demonstration. Replace with your actual model and data.

# Define a simple CNN for demonstration purposes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 classes for MNIST-like data

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        embedding = x # Capture embedding from the second to last layer
        output = self.fc2(x)
        return output, embedding

def setup_demonstration_assets():
    """Creates a dummy model, saves it, and generates a sample input tensor."""
    print("--- Setting up demonstration assets ---")
    
    # Create a dummy model and save its state_dict
    # In your use case, you would already have your `delta_model.pth`
    model_path = "delta_model.pth"
    if not os.path.exists(model_path):
        print(f"Creating dummy '{model_path}' for demonstration...")
        dummy_model = SimpleCNN()
        torch.save(dummy_model.state_dict(), model_path)
    else:
        print(f"Using existing dummy '{model_path}'.")

    # Create a dummy input tensor (e.g., a 28x28 grayscale image)
    # This simulates a single image from a dataset like MNIST.
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Create a small dummy dataset for t-SNE visualization
    # 100 samples, 10 for each of the 10 classes
    data = torch.randn(100, 1, 28, 28)
    labels = torch.repeat_interleave(torch.arange(10), 10)

    print("--- Demonstration assets are ready ---\n")
    return model_path, sample_input, (data, labels)

## -----------------------------------------------------------
## SECTION 1: MODEL LOADING
## -----------------------------------------------------------

def load_model(model_path, model_class):
    """Loads the model from the .pth file."""
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode
    return model

## -----------------------------------------------------------
## SECTION 2: ANALYSIS FUNCTIONS
## -----------------------------------------------------------

# 1. Model Architecture Summary
def analyze_architecture(model, input_size):
    """Prints the model summary using torchsummary."""
    print("\n### 1. Model Architecture Summary ###")
    print("This shows the layers, output shapes, and parameter counts.")
    try:
        summary(model, input_size=input_size)
    except Exception as e:
        print(f"Could not generate summary. Error: {e}")
        print("Printing model object instead:\n", model)

# 2. Weight and Bias Distribution
def analyze_weights(model):
    """Visualizes the distribution of weights and biases for each layer."""
    print("\n### 2. Weight and Bias Distribution ###")
    print("This helps identify issues like vanishing or exploding gradients.")
    fig, axes = plt.subplots(len(list(model.named_parameters())), 1, figsize=(8, 2 * len(list(model.named_parameters()))))
    fig.suptitle('Weight and Bias Distributions per Layer')
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            ax = axes[i]
            ax.hist(param.data.cpu().numpy().flatten(), bins=100)
            ax.set_title(name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 3. Feature Map Visualization
def analyze_feature_maps(model, layer_name, input_tensor):
    """Visualizes the feature maps (activations) of a specific convolutional layer."""
    print(f"\n### 3. Feature Map Visualization (Layer: {layer_name}) ###")
    print("This shows what features the model detects at an intermediate stage.")
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    try:
        # Register the hook
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(get_activation(layer_name))

        # Forward pass
        _ = model(input_tensor)
        handle.remove() # Clean up the hook

        # Plot the feature maps
        acts = activations[layer_name].squeeze()
        num_maps = acts.size(0)
        fig, axes = plt.subplots(num_maps // 4, 4, figsize=(12, 1.5 * (num_maps // 4)))
        fig.suptitle(f'Feature Maps from Layer: {layer_name}')
        for i, ax in enumerate(axes.flat):
            if i < num_maps:
                ax.imshow(acts[i].cpu().numpy(), cmap='viridis')
                ax.set_title(f'Map {i+1}')
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except KeyError:
        print(f"Error: Layer '{layer_name}' not found. Available layers: {list(dict(model.named_modules()).keys())}")
    except Exception as e:
        print(f"Could not visualize feature maps. Error: {e}")


# 4. Saliency Maps
def analyze_saliency(model, input_tensor):
    """Generates a saliency map to show which input pixels are most influential."""
    print("\n### 4. Saliency Maps ###")
    print("This highlights the pixels your model 'looks at' for its prediction.")

    input_tensor.requires_grad_()
    output, _ = model(input_tensor)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    
    output_max.backward()
    
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.detach().squeeze().numpy(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(saliency.cpu().numpy(), cmap='hot')
    plt.title('Saliency Map')
    plt.axis('off')
    plt.suptitle('Saliency Map Analysis')
    plt.show()

# 5. Class Activation Mapping (Grad-CAM)
def analyze_grad_cam(model, target_layer_name, input_tensor):
    """Generates a Grad-CAM to visualize where the model focuses for a decision."""
    print(f"\n### 5. Class Activation Mapping (Grad-CAM) (Layer: {target_layer_name}) ###")
    print("This creates a heatmap over the image to show important regions.")

    try:
        gradients = {}
        activations = {}

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            activations['value'] = output

        target_layer = dict(model.named_modules())[target_layer_name]
        b_handle = target_layer.register_full_backward_hook(backward_hook)
        f_handle = target_layer.register_forward_hook(forward_hook)

        # Forward and backward pass
        output, _ = model(input_tensor)
        output_idx = output.argmax()
        model.zero_grad()
        output[0, output_idx].backward()

        # Get gradients and activations
        grads = gradients['value'].squeeze()
        acts = activations['value'].squeeze()

        b_handle.remove()
        f_handle.remove()
        
        # Pool gradients and compute weights
        weights = torch.mean(grads, dim=[1, 2])
        
        # Compute CAM
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
        
        cam = np.maximum(cam.detach().cpu().numpy(), 0) # ReLU
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Visualize
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_tensor.detach().squeeze().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(input_tensor.detach().squeeze().numpy(), cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title('Grad-CAM Overlay')
        plt.axis('off')
        plt.suptitle('Grad-CAM Analysis')
        plt.show()

    except KeyError:
        print(f"Error: Layer '{target_layer_name}' not found.")
    except Exception as e:
        print(f"Could not generate Grad-CAM. Error: {e}")

# 6. Activation Maximization (Basic Implementation)
def analyze_activation_maximization(model, layer_name, filter_index, input_size=(1, 28, 28)):
    """Generates an image that maximally activates a specific filter."""
    print(f"\n### 6. Activation Maximization (Layer: {layer_name}, Filter: {filter_index}) ###")
    print("This synthetically generates the 'preferred' input for a specific neuron.")
    
    try:
        layer = dict(model.named_modules())[layer_name]
        
        # Create a random noise image
        img = torch.randn(1, *input_size, requires_grad=True)
        optimizer = optim.Adam([img], lr=0.1, weight_decay=1e-6)

        activations = {}
        def hook(model, input, output):
            activations['value'] = output

        handle = layer.register_forward_hook(hook)

        print("Optimizing input image...")
        for i in range(50): # 50 optimization steps
            optimizer.zero_grad()
            _ = model(img)
            
            # Loss is the mean activation of the target filter
            loss = -activations['value'][0, filter_index].mean()
            loss.backward()
            optimizer.step()
        
        handle.remove()
        
        # Visualize the result
        optimized_image = img.detach().squeeze().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(optimized_image, cmap='viridis')
        plt.title(f'Image Maximizing Filter {filter_index} in {layer_name}')
        plt.axis('off')
        plt.show()

    except KeyError:
        print(f"Error: Layer '{layer_name}' not found.")
    except Exception as e:
        print(f"Could not perform activation maximization. Error: {e}")

# 7. Occlusion Sensitivity
def analyze_occlusion(model, input_tensor, patch_size=5):
    """Analyzes model sensitivity by occluding parts of the image."""
    print("\n### 7. Occlusion Sensitivity ###")
    print("This shows which parts of the image are critical for the prediction.")

    output, _ = model(input_tensor)
    original_prob = torch.softmax(output, dim=1).max().item()
    
    _, _, h, w = input_tensor.shape
    heatmap = torch.zeros((h, w))
    
    print("Running occlusion analysis...")
    for i in range(0, h - patch_size + 1):
        for j in range(0, w - patch_size + 1):
            # Create a copy and occlude a patch
            occluded_tensor = input_tensor.clone()
            occluded_tensor[:, :, i:i+patch_size, j:j+patch_size] = 0
            
            # Get new probability
            output, _ = model(occluded_tensor)
            new_prob = torch.softmax(output, dim=1).max().item()
            
            # Heatmap stores the drop in probability
            heatmap[i:i+patch_size, j:j+patch_size] += original_prob - new_prob
            
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.detach().squeeze().numpy(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap.numpy(), cmap='hot')
    plt.title('Occlusion Sensitivity Heatmap')
    plt.axis('off')
    plt.suptitle('Occlusion Sensitivity Analysis')
    plt.show()

# 8. Layer-Wise Relevance Propagation (LRP) - Conceptual
def explain_lrp():
    """Explains LRP as it's complex to implement from scratch."""
    print("\n### 8. Layer-Wise Relevance Propagation (LRP) ###")
    print("LRP is an advanced technique to trace a prediction back to the input pixels.")
    print("It redistributes the output score backward through the network to assign 'relevance' to each neuron.")
    print("Implementing LRP from scratch is complex. It's recommended to use a specialized library like 'captum' for PyTorch.")
    print("Example using captum (not run here):\n")
    print(" from captum.attr import LRP")
    print(" lrp = LRP(model)")
    print(" attribution = lrp.attribute(input_tensor, target=prediction_class_index)")
    print("-" * 50)

# 9. t-SNE Embedding Visualization
def analyze_tsne(model, data, labels):
    """Visualizes the high-dimensional embeddings in 2D using t-SNE."""
    print("\n### 9. t-SNE Embedding Visualization ###")
    print("This shows how well the model separates different classes in its learned feature space.")

    print("Generating embeddings for the dataset...")
    # Get embeddings from the second-to-last layer
    _, embeddings = model(data)
    embeddings = embeddings.detach().cpu().numpy()

    print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels.numpy(), cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
    plt.title('t-SNE Visualization of Model Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# 10. Adversarial Attack Analysis (FGSM)
def analyze_adversarial_attack(model, input_tensor, epsilon=0.1):
    """Generates an adversarial example using the Fast Gradient Sign Method (FGSM)."""
    print("\n### 10. Adversarial Attack Analysis (FGSM) ###")
    print("This tests model robustness by trying to fool it with tiny input perturbations.")

    input_tensor.requires_grad = True
    output, _ = model(input_tensor)
    original_pred = output.argmax(1, keepdim=True).item()
    
    loss = nn.CrossEntropyLoss()(output, torch.tensor([original_pred]))
    model.zero_grad()
    loss.backward()
    
    # Collect the gradient
    data_grad = input_tensor.grad.data
    # Create the perturbed image
    sign_data_grad = data_grad.sign()
    perturbed_image = input_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # Clamp to valid range
    
    # Re-classify the perturbed image
    output_adv, _ = model(perturbed_image)
    adv_pred = output_adv.argmax(1, keepdim=True).item()
    
    print(f"Original Prediction: {original_pred}")
    print(f"Adversarial Prediction: {adv_pred}")

    # Visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_tensor.detach().squeeze().numpy(), cmap='gray')
    plt.title(f'Original (Pred: {original_pred})')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow((epsilon * sign_data_grad).detach().squeeze().numpy(), cmap='gray')
    plt.title(f'Perturbation (Epsilon: {epsilon})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(perturbed_image.detach().squeeze().numpy(), cmap='gray')
    plt.title(f'Adversarial (Pred: {adv_pred})')
    plt.axis('off')
    plt.suptitle('FGSM Adversarial Attack')
    plt.show()

## -----------------------------------------------------------
## SECTION 3: MAIN EXECUTION
## -----------------------------------------------------------

def main():
    """Main function to run all analysis methods."""
    # --- Setup ---
    # This creates a dummy model file and sample data for the script to run.
    # In your real use case, you would provide your own model path and data.
    model_path, sample_input, (tsne_data, tsne_labels) = setup_demonstration_assets()

    # --- Load Your Model ---
    # IMPORTANT: Replace SimpleCNN with your actual model's class.
    # The architecture must match the one saved in 'delta_model.pth'.
    model = load_model(model_path, SimpleCNN)
    
    # --- Run Analysis ---
    # NOTE: Adjust layer names like 'conv2' to match your model's architecture.
    
    # 1. Model Architecture Summary
    analyze_architecture(model, input_size=(1, 28, 28))
    
    # 2. Weight Distribution
    analyze_weights(model)
    
    # 3. Feature Maps (visualizing the output of the second conv layer)
    analyze_feature_maps(model, layer_name='conv2', input_tensor=sample_input)
    
    # 4. Saliency Maps
    analyze_saliency(model, sample_input.clone())
    
    # 5. Grad-CAM (using the final conv layer for best results)
    analyze_grad_cam(model, target_layer_name='conv2', input_tensor=sample_input)

    # 6. Activation Maximization (maximizing filter 5 in the first conv layer)
    analyze_activation_maximization(model, layer_name='conv1', filter_index=5)

    # 7. Occlusion Sensitivity
    analyze_occlusion(model, sample_input)

    # 8. LRP Explanation
    explain_lrp()

    # 9. t-SNE Visualization
    analyze_tsne(model, tsne_data, tsne_labels)

    # 10. Adversarial Attack
    analyze_adversarial_attack(model, sample_input.clone())

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()