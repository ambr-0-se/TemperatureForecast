"""Visualization utilities for weather prediction models."""

import os
import config
import matplotlib.pyplot as plt

def plot_individual_losses(results, save_dir):
    """Plot training and validation loss (mean and std) curves for each model separately."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_type, models in results.items():
        for model_name, metrics in models.items():
            if metrics and 'training_loss' in metrics and 'validation_loss' in metrics:
                training_mean = metrics['training_loss']['mean']
                training_std = metrics['training_loss']['std']
                validation_mean = metrics['validation_loss']['mean']
                validation_std = metrics['validation_loss']['std']

                epochs = range(1, config.epochs + 1)

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, training_mean, label='Training Loss')
                plt.fill_between(epochs, training_mean - training_std, training_mean + training_std, alpha=0.2)
                plt.plot(epochs, validation_mean, label='Validation Loss')
                plt.fill_between(epochs, validation_mean - validation_std, validation_mean + validation_std, alpha=0.2)
                
                plt.xlabel('Epochs')
                plt.ylabel('Loss (MSE)')
                plt.title(f'Training and Validation Loss - {model_type} {model_name}')
                plt.legend()
                plt.grid(True)
                
                plt.savefig(os.path.join(save_dir, f'{model_type}_{model_name}_losses.png'))
                plt.close()

def plot_combined_training_losses(results, save_dir):
    """Plot training losses for all models together."""
    plt.figure(figsize=(10, 6))
    
    for model_type, models in results.items():
        for model_name, metrics in models.items():
            if metrics and 'training_loss' in metrics:
                epochs = range(1, config.epochs + 1)
                plt.plot(epochs, metrics['training_loss']['mean'], 
                        label=f'{model_name}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'combined_training_losses.png'))
    plt.close()

def plot_combined_validation_losses(results, save_dir):
    """Plot validation losses for all models together."""
    plt.figure(figsize=(10, 6))
    
    for model_type, models in results.items():
        for model_name, metrics in models.items():
            if metrics and 'validation_loss' in metrics:
                epochs = range(1, config.epochs + 1)
                plt.plot(epochs, metrics['validation_loss']['mean'], 
                        label=f'{model_name}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'combined_validation_losses.png'))
    plt.close()

def plot_test_results(results, save_dir):
    """Create bar chart of test losses."""
    models = []
    losses = []
    
    for model_type, model_results in results.items():
        for model_name, metrics in model_results.items():
            if metrics and 'test_loss' in metrics:
                models.append(f'{model_name}')
                losses.append(f'{metrics["test_loss"]["mean"]:.4f}')
    
    if models:
        plt.figure(figsize=(10, 6))
        plt.bar(models, losses)
        plt.title('Test Loss Comparison')
        plt.ylabel('MSE Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'test_results.png'))
        plt.close()

def plot_time_results(results, save_dir):
    """Plot time taken for each model."""
    plt.figure(figsize=(10, 6))
    for model_type, model_results in results.items():
        for model_name, metrics in model_results.items():
            if metrics and 'time (in sec)' in metrics:
                plt.plot(model_name, metrics['time (in sec)'], label=model_name)
    plt.xlabel('Model')
    plt.ylabel('Time (in sec)')
    plt.title('Time Comparison')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'time_results.png'))
    plt.close()

def visualize_results(results, save_dir):
    """Generate all visualizations for model results.
    
    Args:
        results: Dictionary containing results for all models
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plot_individual_losses(results, save_dir)
    plot_combined_training_losses(results, save_dir)
    plot_combined_validation_losses(results, save_dir)
    plot_test_results(results, save_dir)
    plot_time_results(results, save_dir)