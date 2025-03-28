# data/dataset.py
"""
PyTorch dataset implementation for transaction data.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import BertTokenizer
import logging

logger = logging.getLogger(__name__)

class TransactionDataset(Dataset):
    """
    PyTorch Dataset for transaction data.
    Prepares transaction descriptions for BERT-based models.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 tokenizer: BertTokenizer,
                 categories: List[str],
                 max_length: int = 128,
                 include_amount: bool = True,
                 include_date_features: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with transaction data
            tokenizer: BERT tokenizer
            categories: List of all possible categories
            max_length: Maximum sequence length for BERT
            include_amount: Whether to include amount as a feature
            include_date_features: Whether to include date-based features
        """
        self.data = data
        self.tokenizer = tokenizer
        self.categories = categories
        self.max_length = max_length
        self.include_amount = include_amount
        self.include_date_features = include_date_features
        
        # Create a mapping from category names to indices
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Prepare category tensors (multi-label format)
        self.category_labels = self._prepare_category_labels()
        
        logger.info(f"Created dataset with {len(data)} samples and {len(categories)} categories")
    
    def _prepare_category_labels(self) -> List[torch.Tensor]:
        """
        Convert category strings to multi-label tensors.
        
        Returns:
            List of tensors, one for each transaction
        """
        labels = []
        for category in self.data['category']:
            # For multi-label classification, we'd normally handle multiple categories
            # But in this case, we only have one category per transaction
            label_idx = self.category_to_idx.get(category, -1)
            if label_idx == -1:
                logger.warning(f"Unknown category: {category}")
                # Create a zero tensor
                label_tensor = torch.zeros(len(self.categories))
            else:
                # Create a one-hot tensor
                label_tensor = torch.zeros(len(self.categories))
                label_tensor[label_idx] = 1.0
            
            labels.append(label_tensor)
        
        return labels
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of transactions
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the transaction
        row = self.data.iloc[idx]
        
        # Get the description (use cleaned if available, otherwise original)
        description = row.get('cleaned_description', row['description'])
        
        # Tokenize the description
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Create the basic item
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': self.category_labels[idx]
        }
        
        # Add amount as a feature if requested
        if self.include_amount:
            # Normalize the amount (simple scaling)
            amount = float(row['amount'])
            # Log-scale normalization (add 1 to avoid log(0))
            normalized_amount = np.log1p(amount) / 10.0  # Rough scaling
            item['amount'] = torch.tensor([normalized_amount], dtype=torch.float)
        
        # Add date features if requested
        if self.include_date_features and 'day_of_week' in row and 'month' in row:
            # One-hot encode day of week (0-6)
            day_of_week = torch.zeros(7)
            day_of_week[row['day_of_week']] = 1.0
            
            # One-hot encode month (1-12)
            month = torch.zeros(12)
            month[row['month'] - 1] = 1.0  # Adjust for 0-indexing
            
            item['day_of_week'] = day_of_week
            item['month'] = month
        
        return item


def create_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: BertTokenizer,
    categories: List[str],
    batch_size: int = 16,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        train_df: Training data
        test_df: Testing data
        tokenizer: BERT tokenizer
        categories: List of categories
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = TransactionDataset(
        train_df, tokenizer, categories, max_length=max_length
    )
    
    test_dataset = TransactionDataset(
        test_df, tokenizer, categories, max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Training batches: {len(train_loader)}, Testing batches: {len(test_loader)}")
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    from transformers import BertTokenizer
    from data.processor import TransactionProcessor
    import tempfile
    import os
    
    # This is just for testing the module directly
    
    # Sample data string (for testing)
    sample_data = '''
    "date","description","amount","category"
    "2024-12-03","COSTCO.CA TRANSACTION",223.05,"Miscellaneous"
    "2025-01-09","BULK BARN 950",191.94,"Groceries"
    "2024-05-28","SWISS CHALET 959",90.95,"Dining"
    "2024-07-11","SCORES 879 PURCHASE",26.0,"Dining"
    "2024-06-24","PRE-AUTHORIZED PAYMENT FEE",412.07,"Banking"
    "2024-11-04","COSTCO GAS 830",48.49,"Retail"
    "2024-12-12","OC TRANSPO 657 TICKET",114.41,"Transportation"
    "2024-04-07","YMCA 533 SERVICE",76.89,"Health"
    "2024-05-19","EBAY*",67.45,"Miscellaneous"
    '''
    
    # Write sample data to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_data)
        temp_file = f.name
    
    try:
        # Load and process the data
        processor = TransactionProcessor()
        df = processor.load_data(temp_file)
        processed_df = processor.extract_features(df)
        
        # Split for training
        train_df, test_df = processor.prepare_for_training(processed_df)
        
        # Get categories
        categories = processor.get_categories()
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            train_df, test_df, tokenizer, categories, batch_size=2
        )
        
        # Print sample batch
        for batch in train_loader:
            print("Sample batch:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: tensor of shape {v.shape}")
                else:
                    print(f"{k}: {v}")
            break
    
    finally:
        # Clean up
        os.unlink(temp_file)
