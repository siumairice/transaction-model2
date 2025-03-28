# models/classifier.py
"""
BERT-based model for transaction categorization.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Dict, Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

class TransactionBertModel(nn.Module):
    """
    BERT-based model for classifying financial transactions.
    Extends BERT with additional features for transaction data.
    """
    
    def __init__(
        self, 
        num_labels: int,
        model_name: str = 'bert-base-uncased',
        include_amount: bool = True,
        include_date_features: bool = True
    ):
        """
        Initialize the transaction classifier model.
        
        Args:
            num_labels: Number of classification categories
            model_name: Pre-trained BERT model name
            include_amount: Whether to include transaction amount as a feature
            include_date_features: Whether to include date-based features
        """
        super().__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_config = self.bert.config
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        
        # Track feature flags
        self.include_amount = include_amount
        self.include_date_features = include_date_features
        
        # Calculate input size for the classifier
        classifier_input_size = self.bert_config.hidden_size
        
        # Add additional feature inputs if needed
        if include_amount:
            self.amount_projection = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            classifier_input_size += 32
        
        if include_date_features:
            # Day of week (7 dims) + month (12 dims)
            self.date_projection = nn.Sequential(
                nn.Linear(19, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            classifier_input_size += 32
        
        # Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
        
        logger.info(f"Initialized TransactionBertModel with {num_labels} labels")
        logger.info(f"Input size for classifier: {classifier_input_size}")
    
    def _init_weights(self, module):
        """Initialize the weights of new modules."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        amount: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            amount: Transaction amounts (batch_size, 1) (optional)
            day_of_week: One-hot day of week (batch_size, 7) (optional)
            month: One-hot month (batch_size, 12) (optional)
            labels: Classification labels (batch_size, num_labels) (optional)
            
        Returns:
            Dictionary with model outputs
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # List to collect all features
        features = [pooled_output]
        
        # Add amount feature if provided
        if self.include_amount and amount is not None:
            amount_features = self.amount_projection(amount)
            features.append(amount_features)
        
        # Add date features if provided
        if self.include_date_features and day_of_week is not None and month is not None:
            # Concatenate date features
            date_features = torch.cat([day_of_week, month], dim=1)
            date_features = self.date_projection(date_features)
            features.append(date_features)
        
        # Combine all features
        combined_features = torch.cat(features, dim=1)
        
        # Get logits
        logits = self.classifier(combined_features)
        
        # Prepare output dictionary
        output_dict = {
            'logits': logits,
            'features': combined_features
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            if len(labels.shape) == 1:  # Single-label classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:  # Multi-label classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            
            output_dict['loss'] = loss
        
        return output_dict
    
    def save_pretrained(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save BERT component
        self.bert.save_pretrained(os.path.join(output_dir, "bert"))
        
        # Save model config
        config = {
            'num_labels': self.classifier.out_features,
            'include_amount': self.include_amount,
            'include_date_features': self.include_date_features,
        }
        
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f)
        
        # Save full model state
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir: str, device: Optional[torch.device] = None):
        """
        Load a model from the specified directory.
        
        Args:
            model_dir: Directory containing the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        import os
        import json
        
        # Load config
        with open(os.path.join(model_dir, "config.json"), 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            num_labels=config['num_labels'],
            model_name=os.path.join(model_dir, "bert"),
            include_amount=config.get('include_amount', True),
            include_date_features=config.get('include_date_features', True)
        )
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=device if device else torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        
        # Move to device if specified
        if device:
            model = model.to(device)
        
        logger.info(f"Model loaded from {model_dir}")
        return model


def create_transaction_model(
    num_categories: int, 
    model_name: str = 'bert-base-uncased',
    include_amount: bool = True,
    include_date_features: bool = True,
    device: Optional[torch.device] = None
) -> TransactionBertModel:
    """
    Create a transaction classification model.
    
    Args:
        num_categories: Number of transaction categories
        model_name: Pre-trained model name
        include_amount: Whether to include amount features
        include_date_features: Whether to include date features
        device: Device to load the model to
        
    Returns:
        Transaction classification model
    """
    model = TransactionBertModel(
        num_labels=num_categories,
        model_name=model_name,
        include_amount=include_amount,
        include_date_features=include_date_features
    )
    
    if device:
        model = model.to(device)
        
    return model
