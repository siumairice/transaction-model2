# utils/metrics.py
"""
Evaluation metrics for transaction classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

def evaluate_predictions(
    true_labels: Union[List[str], np.ndarray],
    predicted_labels: Union[List[str], np.ndarray],
    categories: List[str],
    labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate predictions with various metrics.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        categories: List of all possible categories
        labels: Optional list of labels to use (defaults to all found in the data)
        output_dir: Directory to save evaluation results (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert labels to arrays if they're lists
    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    if isinstance(predicted_labels, list):
        predicted_labels = np.array(predicted_labels)
    
    # If labels not provided, get unique labels from data
    if labels is None:
        labels = sorted(np.unique(np.concatenate([true_labels, predicted_labels])))
        
    # Ensure zero_division=0 to handle categories without samples gracefully
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'macro_precision': precision_score(true_labels, predicted_labels, average='macro', 
                                         zero_division=0, labels=labels),
        'macro_recall': recall_score(true_labels, predicted_labels, average='macro', 
                                   zero_division=0, labels=labels),
        'macro_f1': f1_score(true_labels, predicted_labels, average='macro', 
                           zero_division=0, labels=labels),
        'weighted_precision': precision_score(true_labels, predicted_labels, average='weighted', 
                                            zero_division=0, labels=labels),
        'weighted_recall': recall_score(true_labels, predicted_labels, average='weighted', 
                                      zero_division=0, labels=labels),
        'weighted_f1': f1_score(true_labels, predicted_labels, average='weighted', 
                              zero_division=0, labels=labels)
    }
    
    # Print metrics
    logger.info("Classification Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    # Create classification report
    try:
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=categories, labels=labels)
        logger.info(f"Classification Report:\n{report}")
    except ValueError as e:
        logger.warning(f"Could not generate classification report: {str(e)}")
        # Fall back to a simpler report format
        logger.info("Simple per-category metrics:")
        for i, label in enumerate(labels):
            label_metrics = {
                'precision': precision_score(true_labels, predicted_labels, 
                                          labels=[label], average='micro', zero_division=0),
                'recall': recall_score(true_labels, predicted_labels, 
                                    labels=[label], average='micro', zero_division=0),
                'f1': f1_score(true_labels, predicted_labels, 
                             labels=[label], average='micro', zero_division=0)
            }
            logger.info(f"{label}: {label_metrics}")
    
    # Save results if output_dir is provided
    if output_dir:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        metrics_df.to_csv(f"{output_dir}/metrics_{timestamp}.csv", index=False)
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png")
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics

def calculate_multilabel_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate metrics for multi-label classification.
    
    Args:
        y_true: True binary labels (batch_size, num_classes)
        y_pred: Predicted probabilities (batch_size, num_classes)
        threshold: Threshold for positive prediction
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy for metric calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    # Samples dimension
    accuracy_samples = (y_true == y_pred_binary).mean(axis=1).mean()
    
    # Calculate metrics per class and average
    precision = []
    recall = []
    f1 = []
    
    for i in range(y_true.shape[1]):
        # Skip classes with no positive examples in this batch
        if y_true[:, i].sum() == 0:
            continue
        
        # True positives, false positives, false negatives
        tp = ((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1)).sum()
        fn = ((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0)).sum()
        
        # Calculate metrics for this class
        if tp + fp == 0:  # Avoid division by zero
            class_precision = 0
        else:
            class_precision = tp / (tp + fp)
        
        if tp + fn == 0:  # Avoid division by zero
            class_recall = 0
        else:
            class_recall = tp / (tp + fn)
        
        if class_precision + class_recall == 0:  # Avoid division by zero
            class_f1 = 0
        else:
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
    
    # Average metrics
    avg_precision = np.mean(precision) if precision else 0
    avg_recall = np.mean(recall) if recall else 0
    avg_f1 = np.mean(f1) if f1 else 0
    
    return {
        'accuracy': accuracy_samples,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with lists of metrics from training
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
    
    # Plot training and validation accuracy
    if 'train_accuracy' in history and 'val_accuracy' in history:
        plt.subplot(2, 2, 2)
        plt.plot(history['train_accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    
    # Plot F1 score if available
    if 'train_f1' in history and 'val_f1' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['train_f1'], label='Training F1')
        plt.plot(history['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
    
    # Plot precision and recall if available
    if 'val_precision' in history and 'val_recall' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['val_precision'], label='Validation Precision')
        plt.plot(history['val_recall'], label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Precision and Recall')
        plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Training history plot saved to {output_path}")
    else:
        plt.show()

def get_confusion_by_amount(
    true_labels: List[str],
    predicted_labels: List[str],
    amounts: List[float],
    categories: List[str],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze confusion patterns by transaction amount.
    
    Args:
        true_labels: Ground truth category labels
        predicted_labels: Predicted category labels
        amounts: Transaction amounts
        categories: List of all possible categories
        output_path: Path to save analysis (optional)
        
    Returns:
        DataFrame with confusion analysis by amount
    """
    # Create a DataFrame with predictions and amounts
    df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'amount': amounts,
        'correct': [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    
    # Group by amount bins
    df['amount_bin'] = pd.cut(
        df['amount'],
        bins=[0, 10, 25, 50, 100, 250, 500, 1000, float('inf')],
        labels=['0-10', '10-25', '25-50', '50-100', '100-250', '250-500', '500-1000', '1000+']
    )
    
    # Accuracy by amount bin
    accuracy_by_amount = df.groupby('amount_bin')['correct'].mean()
    
    # Most common confusions by amount bin
    confusion_by_amount = []
    
    for amount_bin, group in df[~df['correct']].groupby('amount_bin'):
        # Count confusion patterns
        confusion_counts = group.groupby(['true_label', 'predicted_label']).size().reset_index()
        confusion_counts.columns = ['True', 'Predicted', 'Count']
        confusion_counts = confusion_counts.sort_values('Count', ascending=False)
        
        # Add to results
        for _, row in confusion_counts.head(3).iterrows():
            confusion_by_amount.append({
                'Amount Bin': amount_bin,
                'True Label': row['True'],
                'Predicted Label': row['Predicted'],
                'Count': row['Count']
            })
    
    confusion_df = pd.DataFrame(confusion_by_amount)
    
    # Plot accuracy by amount
    plt.figure(figsize=(10, 6))
    accuracy_by_amount.plot(kind='bar')
    plt.title('Accuracy by Transaction Amount')
    plt.xlabel('Amount Range')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f"{output_path}_accuracy_by_amount.png")
        confusion_df.to_csv(f"{output_path}_confusion_by_amount.csv", index=False)
        logger.info(f"Amount analysis saved to {output_path}")
    
    return confusion_df


# Example usage
if __name__ == "__main__":
    # Sample data for testing
    true_cats = ['Dining', 'Groceries', 'Dining', 'Transportation', 'Retail', 'Health']
    pred_cats = ['Dining', 'Groceries', 'Retail', 'Transportation', 'Retail', 'Miscellaneous']
    categories = ['Dining', 'Groceries', 'Transportation', 'Retail', 'Health', 'Miscellaneous']
    amounts = [25.50, 120.75, 35.20, 15.00, 75.50, 45.25]
    
    # Calculate metrics
    metrics = evaluate_predictions(true_cats, pred_cats, categories)
    
    # Get confusion by amount
    confusion_df = get_confusion_by_amount(true_cats, pred_cats, amounts, categories)
    print("\nConfusion patterns by amount:")
    print(confusion_df)
    
    # Test multi-label metrics
    y_true = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float)
    
    y_pred = torch.tensor([
        [0.9, 0.1, 0.2],
        [0.2, 0.8, 0.1],
        [0.1, 0.4, 0.7],
        [0.7, 0.6, 0.1]
    ], dtype=torch.float)
    
    ml_metrics = calculate_multilabel_metrics(y_true, y_pred)
    print("\nMulti-label metrics:")
    for metric, value in ml_metrics.items():
        print(f"{metric}: {value:.4f}")
