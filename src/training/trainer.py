# training/trainer.py
"""
Training functionality for transaction categorization models.
"""

import os
import json
import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from models.classifier import TransactionBertModel
from utils.metrics import evaluate_predictions, calculate_multilabel_metrics

logger = logging.getLogger(__name__)

class TransactionModelTrainer:
    """
    Trainer class for transaction categorization models.
    Handles the training loop, evaluation, early stopping, and model saving.
    """
    
    def __init__(
        self,
        model: TransactionBertModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        id_to_category: Dict[int, str],
        output_dir: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 4,
        warmup_proportion: float = 0.1,
        early_stopping_patience: int = 2,
        device: Optional[torch.device] = None,
        multi_label: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The transaction classification model
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            id_to_category: Mapping from label IDs to category names
            output_dir: Directory to save model checkpoints and results
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of training epochs
            warmup_proportion: Proportion of training steps for LR warmup
            early_stopping_patience: Number of epochs to wait before stopping
            device: Device to train on (cpu or cuda)
            multi_label: Whether to use multi-label classification
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.id_to_category = id_to_category
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_proportion = warmup_proportion
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_label = multi_label
        
        # Move model to device
        self.model.to(self.device)
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_proportion)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize tracking variables
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.best_model_path = os.path.join(output_dir, "best_model")
        
        logger.info(f"TransactionModelTrainer initialized")
        logger.info(f"Training device: {self.device}")
        logger.info(f"Number of training batches: {len(train_dataloader)}")
        logger.info(f"Number of validation batches: {len(val_dataloader)}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_metrics['loss']:.4f}")
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                logger.info(f"Saving best model to {self.best_model_path}")
                self.save_model(self.best_model_path)
            else:
                self.patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc="Training", 
            leave=False,
            disable=False
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                amount=batch.get('amount'),
                day_of_week=batch.get('day_of_week'),
                month=batch.get('month'),
                labels=batch['labels']
            )
            
            # Get loss
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            self.val_dataloader, 
            desc="Validation", 
            leave=False,
            disable=False
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    amount=batch.get('amount'),
                    day_of_week=batch.get('day_of_week'),
                    month=batch.get('month'),
                    labels=batch['labels']
                )
                
                # Get loss and predictions
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Process predictions based on whether we're doing multi-label or single-label
                if self.multi_label:
                    # Multi-label: apply sigmoid and threshold
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    all_preds.append(preds.cpu())
                    all_labels.append(batch['labels'].cpu())
                else:
                    # Single-label: argmax
                    preds = torch.argmax(logits, dim=1)
                    # For single-label, we need to convert one-hot encoded labels to class indices
                    if batch['labels'].dim() > 1 and batch['labels'].shape[1] > 1:
                        labels = torch.argmax(batch['labels'], dim=1)
                    else:
                        labels = batch['labels']
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
        
        # Compute metrics
        if self.multi_label:
            # Combine all predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate multi-label metrics
            metrics = calculate_multilabel_metrics(all_labels, all_preds)
            metrics['loss'] = total_loss / len(self.val_dataloader)
            
        else:
            # Single-label classification
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            
            accuracy = (all_preds == all_labels).mean()
            
            # Convert to category names for detailed metrics
            pred_categories = [self.id_to_category[int(pred)] for pred in all_preds]
            true_categories = [self.id_to_category[int(label)] for label in all_labels]
            
            # Calculate metrics using our utility function
            categories = list(self.id_to_category.values())
            detailed_metrics = evaluate_predictions(
                true_categories, 
                pred_categories, 
                categories,
                output_dir=None  # Don't save plots during training
            )
            
            metrics = {
                'loss': total_loss / len(self.val_dataloader),
                'accuracy': accuracy,
                'precision': detailed_metrics['macro_precision'],
                'recall': detailed_metrics['macro_recall'],
                'f1': detailed_metrics['macro_f1']
            }
        
        return metrics
    
    def save_model(self, output_path: str) -> None:
        """
        Save the model, tokenizer, and metadata.
        
        Args:
            output_path: Path to save the model
        """
        # Save the model
        self.model.save_pretrained(output_path)
        
        # Save the tokenizer
        # We need to get the tokenizer from the dataset
        tokenizer = self.train_dataloader.dataset.tokenizer
        tokenizer.save_pretrained(output_path)
        
        # Save category mapping
        with open(os.path.join(output_path, "category_mapping.json"), 'w') as f:
            # Convert int keys to strings for JSON serialization
            category_mapping = {str(k): v for k, v in self.id_to_category.items()}
            json.dump(category_mapping, f)
        
        # Save training metadata
        metadata = {
            'multi_label': self.multi_label,
            'num_categories': len(self.id_to_category),
            'categories': list(self.id_to_category.values()),
            'training_params': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_proportion': self.warmup_proportion,
                'early_stopping_patience': self.early_stopping_patience,
            },
            'best_metrics': {
                'val_loss': self.best_val_loss,
                'val_f1': self.best_val_f1,
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_path, "training_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_history(self) -> None:
        """Save the training history to disk."""
        # Convert to DataFrame for easier analysis
        history_df = pd.DataFrame(self.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        
        # Save as CSV
        history_path = os.path.join(self.output_dir, "training_history.csv")
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")
        
        # Plot and save graphs
        self._plot_training_history()
    
    def _plot_training_history(self) -> None:
        """Create and save training history plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Convert history to DataFrame
            history_df = pd.DataFrame(self.history)
            history_df['epoch'] = range(1, len(history_df) + 1)
            
            # Loss plot
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
            plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Metrics plot
            plt.subplot(1, 2, 2)
            plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Accuracy')
            plt.plot(history_df['epoch'], history_df['val_precision'], label='Precision')
            plt.plot(history_df['epoch'], history_df['val_recall'], label='Recall')
            plt.plot(history_df['epoch'], history_df['val_f1'], label='F1')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Metrics')
            plt.legend()
            
            plt.tight_layout()
            
            # Save the figure
            plot_path = os.path.join(self.output_dir, "training_metrics.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Training plots saved to {plot_path}")
            
            # Learning rate plot
            plt.figure(figsize=(8, 4))
            plt.plot(history_df['epoch'], history_df['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            
            # Save the figure
            lr_plot_path = os.path.join(self.output_dir, "learning_rate.png")
            plt.savefig(lr_plot_path)
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping history plots.")
        except Exception as e:
            logger.error(f"Error creating history plots: {str(e)}")


def train_transaction_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    id_to_category: Dict[int, str],
    output_dir: str,
    num_categories: int,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    include_amount: bool = True,
    include_date_features: bool = True,
    multi_label: bool = False
) -> Tuple[TransactionBertModel, Dict[str, List[float]]]:
    """
    Train a transaction classification model.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        id_to_category: Mapping from label IDs to category names
        output_dir: Directory to save model and results
        num_categories: Number of transaction categories
        learning_rate: Learning rate for optimizer
        num_epochs: Maximum number of training epochs
        include_amount: Whether to include amount features
        include_date_features: Whether to include date features
        multi_label: Whether to use multi-label classification
        
    Returns:
        Tuple of (trained model, training history)
    """
    from models.classifier import create_transaction_model
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_transaction_model(
        num_categories=num_categories,
        include_amount=include_amount,
        include_date_features=include_date_features,
        device=device
    )
    
    # Create trainer
    trainer = TransactionModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        id_to_category=id_to_category,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        multi_label=multi_label
    )
    
    # Train the model
    history = trainer.train()
    
    # Load the best model
    best_model = TransactionBertModel.from_pretrained(
        os.path.join(output_dir, "best_model"),
        device=device
    )
    
    return best_model, history
