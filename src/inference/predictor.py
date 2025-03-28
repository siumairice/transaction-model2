# inference/predictor.py
"""
Inference functionality for transaction categorization models.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import BertTokenizer
import logging
from tqdm import tqdm

from models.classifier import TransactionBertModel
from data.processor import TransactionProcessor

logger = logging.getLogger(__name__)

class TransactionPredictor:
    """
    Predictor class for making transaction categorization predictions.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        max_length: int = 128
    ):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_dir: Directory containing the trained model
            device: Device to run inference on
            batch_size: Batch size for inference
            max_length: Maximum sequence length for BERT
        """
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # Load the model
        self.model = TransactionBertModel.from_pretrained(model_dir, device=self.device)
        self.model.eval()
        
        # Load category mapping
        category_path = os.path.join(model_dir, "category_mapping.json")
        if os.path.exists(category_path):
            with open(category_path, 'r') as f:
                self.id_to_category = json.load(f)
                # Convert string keys back to integers
                self.id_to_category = {int(k): v for k, v in self.id_to_category.items()}
        else:
            logger.warning(f"Category mapping not found at {category_path}")
            self.id_to_category = {}
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.multi_label = self.metadata.get('multi_label', False)
        else:
            logger.warning(f"Metadata not found at {metadata_path}")
            self.metadata = {}
            self.multi_label = False
        
        # Create category to ID mapping for reverse lookup
        self.category_to_id = {v: k for k, v in self.id_to_category.items()}
        
        logger.info(f"TransactionPredictor initialized from {model_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Multi-label: {self.multi_label}")
        logger.info(f"Number of categories: {len(self.id_to_category)}")
    
    def predict(
        self, 
        transactions: Union[List[str], pd.DataFrame],
        include_amounts: bool = False,
        include_dates: bool = False,
        threshold: float = 0.08,  # Lower default threshold based on your data
        return_confidences: bool = True
    ) -> pd.DataFrame:
        """
        Predict categories for transaction descriptions.
        
        Args:
            transactions: List of transaction descriptions or DataFrame with transactions
            include_amounts: Whether to include transaction amounts in predictions
            include_dates: Whether to include transaction dates in predictions
            threshold: Confidence threshold for returning multiple categories
            return_confidences: Whether to return prediction confidences
            
        Returns:
            DataFrame with predictions
        """
        # Prepare inputs
        if isinstance(transactions, list):
            # Convert list to DataFrame
            df = pd.DataFrame({'description': transactions})
        else:
            # Make a copy to avoid modifying the input
            df = transactions.copy()
        
        # Process transaction data
        processor = TransactionProcessor()
        
        # If we have actual transaction data (rather than just descriptions)
        if isinstance(transactions, pd.DataFrame):
            if all(col in df.columns for col in ['date', 'description', 'amount']):
                # Process the data to extract features
                df = processor.extract_features(df)
        else:
            # Clean descriptions if we only have text
            df['cleaned_description'] = df['description'].apply(
                lambda x: processor.clean_description(x)
            )
            
        # Run predictions
        results = []
        
        # Process in batches
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i+self.batch_size]
            batch_results = self._predict_batch(
                batch_df, 
                include_amounts=include_amounts,
                include_dates=include_dates, 
                threshold=threshold,
                return_confidences=return_confidences
            )
            results.extend(batch_results)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add original data
        if 'description' in df.columns:
            results_df['original_description'] = df['description'].values
        
        return results_df
    
    def _predict_batch(
        self, 
        batch_df: pd.DataFrame,
        include_amounts: bool,
        include_dates: bool,
        threshold: float,
        return_confidences: bool
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict categories for a batch of transactions.
        
        Args:
            batch_df: DataFrame with batch of transactions
            include_amounts: Whether to include amounts
            include_dates: Whether to include dates
            threshold: Threshold for multi-label classification
            return_confidences: Whether to return confidences
            
        Returns:
            List of dictionaries with predictions
        """
        # Determine which field to use for prediction
        text_field = 'cleaned_description' if 'cleaned_description' in batch_df.columns else 'description'
        
        # Tokenize descriptions
        texts = batch_df[text_field].tolist()
        
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Prepare additional features if needed
        amount = None
        day_of_week = None
        month = None
        
        if include_amounts and 'amount' in batch_df.columns:
            # Normalize and convert amounts
            amounts = batch_df['amount'].astype(float).values
            amounts = np.log1p(np.abs(amounts)) / 10.0  # Simple normalization
            amount = torch.tensor(amounts, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        if include_dates and all(col in batch_df.columns for col in ['day_of_week', 'month']):
            # One-hot encode day of week and month
            days = np.zeros((len(batch_df), 7))
            months = np.zeros((len(batch_df), 12))
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                days[i, row['day_of_week']] = 1
                months[i, row['month'] - 1] = 1  # Month is 1-indexed
            
            day_of_week = torch.tensor(days, dtype=torch.float32).to(self.device)
            month = torch.tensor(months, dtype=torch.float32).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                amount=amount,
                day_of_week=day_of_week,
                month=month
            )
        
        logits = outputs['logits']
        
        # Process based on multi-label or single-label
        results = []
        
        if self.multi_label:
            # Multi-label: apply sigmoid and threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions = (probs > threshold).astype(int)
            
            for i in range(len(batch_df)):
                result = {'description': texts[i]}
                
                # Get categories where prediction is 1
                predicted_categories = []
                confidences = []
                
                for j in range(len(probs[i])):
                    if predictions[i, j] == 1:
                        cat_name = self.id_to_category.get(j, f"Category_{j}")
                        predicted_categories.append(cat_name)
                        confidences.append(float(probs[i, j]))
                
                result['predicted_categories'] = predicted_categories
                
                if return_confidences:
                    result['confidences'] = confidences
                
                results.append(result)
        else:
            # Single-label but return multiple if above threshold
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(batch_df)):
                result = {'description': texts[i]}
                
                # Get all categories above threshold
                above_threshold = []
                above_threshold_confidences = []
                
                for j in range(len(probs[i])):
                    if probs[i, j] > threshold:
                        cat_name = self.id_to_category.get(int(j), f"Category_{j}")
                        above_threshold.append(cat_name)
                        above_threshold_confidences.append(float(probs[i, j]))
                
                # Sort by confidence (highest first)
                if above_threshold:
                    sorted_indices = np.argsort(above_threshold_confidences)[::-1]
                    above_threshold = [above_threshold[j] for j in sorted_indices]
                    above_threshold_confidences = [above_threshold_confidences[j] for j in sorted_indices]
                
                # Still include the predicted_category for backward compatibility
                pred_idx = np.argmax(probs[i])
                result['predicted_category'] = self.id_to_category.get(int(pred_idx), f"Category_{pred_idx}")
                
                if return_confidences:
                    result['confidence'] = float(probs[i, pred_idx])
                
                # Add the new multi-category output
                result['above_threshold_categories'] = above_threshold
                if return_confidences:
                    result['above_threshold_confidences'] = above_threshold_confidences
                
                # Print for debugging
                print(f"Transaction: {texts[i]}")
                for j in range(len(probs[i])):
                    cat_name = self.id_to_category.get(int(j), f"Category_{j}")
                    conf = float(probs[i, j])
                    if conf > 0.01:  # Only show non-negligible confidences
                        print(f"  {cat_name}: {conf:.4f}")
                print(f"  Above threshold: {above_threshold}")
                print()
                
                results.append(result)
        
        return results
    
    def predict_csv(
        self, 
        input_file: str,
        output_file: str,
        description_col: str = 'description',
        amount_col: Optional[str] = 'amount',
        date_col: Optional[str] = 'date',
        threshold: float = 0.08  # Lower threshold based on your data
    ) -> pd.DataFrame:
        """
        Predict categories for transactions in a CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            description_col: Column name for transaction descriptions
            amount_col: Column name for amounts (or None to ignore)
            date_col: Column name for dates (or None to ignore)
            threshold: Confidence threshold for returning multiple categories
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Predicting categories for transactions in {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)
        
        # Verify columns
        if description_col not in df.columns:
            raise ValueError(f"Description column '{description_col}' not found in CSV")
        
        # Rename columns to match expected format
        df_renamed = df.copy()
        df_renamed.rename(columns={description_col: 'description'}, inplace=True)
        
        if amount_col and amount_col in df.columns:
            df_renamed.rename(columns={amount_col: 'amount'}, inplace=True)
            include_amounts = True
        else:
            include_amounts = False
        
        if date_col and date_col in df.columns:
            df_renamed.rename(columns={date_col: 'date'}, inplace=True)
            include_dates = True
        else:
            include_dates = False
        
        # Make predictions
        predictions = self.predict(
            df_renamed,
            include_amounts=include_amounts,
            include_dates=include_dates,
            threshold=threshold,
            return_confidences=True
        )
        
        # Create a new DataFrame to store the results
        result_df = df.copy()
        
        # Add predicted category and confidence
        result_df['predicted_category'] = predictions['predicted_category']
        result_df['confidence'] = predictions['confidence']
        
        # Add categories above threshold
        categories_above_threshold = []
        for i, row in predictions.iterrows():
            if 'above_threshold_categories' in row and isinstance(row['above_threshold_categories'], list):
                # Format with confidences
                formatted_cats = []
                for cat, conf in zip(row['above_threshold_categories'], row['above_threshold_confidences']):
                    formatted_cats.append(f"{cat} ({conf:.2f})")
                categories_above_threshold.append(', '.join(formatted_cats))
            else:
                categories_above_threshold.append('')
        
        result_df['categories_above_threshold'] = categories_above_threshold
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        return result_df
    
    def evaluate(
        self, 
        df: pd.DataFrame,
        description_col: str = 'description',
        true_category_col: str = 'category',
        amount_col: Optional[str] = 'amount',
        date_col: Optional[str] = 'date',
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Args:
            df: DataFrame with transactions and true categories
            description_col: Column name for descriptions
            true_category_col: Column name for true categories
            amount_col: Column name for amounts (or None to ignore)
            date_col: Column name for dates (or None to ignore)
            output_dir: Directory to save evaluation results (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        from utils.metrics import evaluate_predictions, calculate_multilabel_metrics
        
        # Verify columns
        if description_col not in df.columns:
            raise ValueError(f"Description column '{description_col}' not found")
        if true_category_col not in df.columns:
            raise ValueError(f"True category column '{true_category_col}' not found")
        
        # Rename columns to match expected format
        df_renamed = df.copy()
        df_renamed.rename(columns={
            description_col: 'description',
            true_category_col: 'category'
        }, inplace=True)
        
        if amount_col and amount_col in df.columns:
            df_renamed.rename(columns={amount_col: 'amount'}, inplace=True)
            include_amounts = True
        else:
            include_amounts = False
        
        if date_col and date_col in df.columns:
            df_renamed.rename(columns={date_col: 'date'}, inplace=True)
            include_dates = True
        else:
            include_dates = False
        
        # Make predictions
        predictions = self.predict(
            df_renamed,
            include_amounts=include_amounts,
            include_dates=include_dates,
            return_confidences=False
        )
        
        # Get true and predicted categories
        true_categories = df_renamed['category'].tolist()
        
        if self.multi_label:
            # Handle multi-label format
            # Convert true categories to multi-hot encoding
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            
            # True labels may be in different formats
            if isinstance(true_categories[0], str):
                # Assume comma-separated values
                true_labels = [cats.split(',') for cats in true_categories]
            else:
                true_labels = true_categories
                
            y_true = mlb.fit_transform(true_labels)
            
            # Extract predicted labels
            pred_labels = predictions['predicted_categories'].tolist()
            y_pred = mlb.transform(pred_labels)
            
            # Calculate metrics
            metrics = calculate_multilabel_metrics(
                torch.tensor(y_true),
                torch.tensor(y_pred)
            )
        else:
            # Single-label format
            pred_categories = predictions['predicted_category'].tolist()
            
            # Get all unique categories
            all_categories = list(set(true_categories + pred_categories))
            
            # Calculate metrics
            metrics = evaluate_predictions(
                true_categories,
                pred_categories,
                all_categories,
                output_dir=output_dir
            )
        
        return metrics

# Function for easy batch prediction
def predict_transactions(
    model_dir: str,
    transactions: Union[List[str], pd.DataFrame],
    include_amounts: bool = False,
    include_dates: bool = False,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    threshold: float = 0.08  # Lower threshold based on your data
) -> pd.DataFrame:
    """
    Predict categories for a list of transaction descriptions.
    
    Args:
        model_dir: Directory with trained model
        transactions: List of transaction descriptions or DataFrame
        include_amounts: Whether to include amounts
        include_dates: Whether to include dates
        batch_size: Batch size for inference
        device: Device to run inference on
        threshold: Confidence threshold for returning multiple categories
        
    Returns:
        DataFrame with predictions
    """
    predictor = TransactionPredictor(
        model_dir=model_dir,
        batch_size=batch_size,
        device=device
    )
    
    return predictor.predict(
        transactions=transactions,
        include_amounts=include_amounts,
        include_dates=include_dates,
        threshold=threshold,
        return_confidences=True
    )