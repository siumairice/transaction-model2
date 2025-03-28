# data/processor.py
"""
Data processing utilities for transaction categorization.
Handles ingestion, cleaning, and preparation of transaction data.
"""

import os
import re
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionProcessor:
    """
    Handles processing of transaction data for the categorization model.
    """
    
    def __init__(self, 
                 remove_numbers: bool = True,
                 remove_special_chars: bool = True,
                 lowercase: bool = True):
        """
        Initialize the transaction processor.
        
        Args:
            remove_numbers: Whether to remove numbers from descriptions
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert text to lowercase
        """
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.categories = set()
        self.merchant_patterns = {}  # For future merchant standardization
        
        # Common words to remove (transaction noise)
        self.noise_words = {
            'transaction', 'purchase', 'payment', 'online', 'store', 'shop',
            'ltd', 'inc', 'limited', 'corporation', 'llc', 'fee', 'service'
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load transaction data from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing transaction data
            
        Returns:
            Pandas DataFrame with the transaction data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transaction data file not found: {file_path}")
        
        try:
            # Try to load with pandas first (handles most CSV variations)
            df = pd.read_csv(file_path)
            
            # Check if we have the expected columns
            required_columns = {'date', 'description', 'amount', 'category'}
            actual_columns = set(df.columns)
            
            if not required_columns.issubset(actual_columns):
                missing = required_columns - actual_columns
                raise ValueError(f"Missing required columns: {missing}")
            
            logger.info(f"Successfully loaded {len(df)} transactions from {file_path}")
            
            # Update our set of categories
            self.categories.update(df['category'].unique())
            
            return df
            
        except Exception as e:
            # If pandas fails, try a more manual approach
            logger.warning(f"Standard CSV loading failed, trying manual approach: {str(e)}")
            
            transactions = []
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Normalize header names
                header = [col.lower().strip() for col in header]
                
                # Find the indices for our required columns
                try:
                    date_idx = header.index('date')
                    desc_idx = header.index('description')
                    amount_idx = header.index('amount')
                    category_idx = header.index('category')
                except ValueError as ve:
                    raise ValueError(f"Could not find required columns in CSV: {str(ve)}")
                
                # Read all transactions
                for row in reader:
                    if len(row) >= max(date_idx, desc_idx, amount_idx, category_idx) + 1:
                        transactions.append({
                            'date': row[date_idx],
                            'description': row[desc_idx],
                            'amount': row[amount_idx],
                            'category': row[category_idx]
                        })
                        # Update categories
                        self.categories.add(row[category_idx])
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            logger.info(f"Manually loaded {len(df)} transactions from {file_path}")
            return df
    
    def clean_description(self, description: str) -> str:
        """
        Clean a transaction description by removing noise.
        
        Args:
            description: Raw transaction description
            
        Returns:
            Cleaned description
        """
        if not description:
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            description = description.lower()
        
        # Remove numbers if specified
        if self.remove_numbers:
            description = re.sub(r'\d+', ' ', description)
        
        # Remove special characters if specified
        if self.remove_special_chars:
            description = re.sub(r'[^\w\s]', ' ', description)
        
        # Remove multiple spaces
        description = re.sub(r'\s+', ' ', description)
        
        # Remove common noise words
        words = description.split()
        filtered_words = [word for word in words if word.lower() not in self.noise_words]
        
        return ' '.join(filtered_words).strip()
        # Enhanced feature extraction for Canadian transactions

    def extract_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced features for Canadian transaction data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with additional extracted features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Clean descriptions
        result['cleaned_description'] = result['description'].apply(clean_transaction_text)
        
        # Convert dates to datetime
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        
        # Extract date features
        result['day_of_week'] = result['date'].dt.dayofweek
        result['month'] = result['date'].dt.month
        result['day'] = result['date'].dt.day
        result['is_weekend'] = result['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        result['is_month_start'] = result['day'].apply(lambda x: 1 if x <= 5 else 0)
        result['is_month_end'] = result['day'].apply(lambda x: 1 if x >= 25 else 0)
        
        # Canadian-specific: extract season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:  # 9, 10, 11
                return 'fall'
                
        result['season'] = result['month'].apply(get_season)
        
        # Convert amount to numeric
        result['amount'] = pd.to_numeric(result['amount'], errors='coerce')
        
        # Create amount bins (useful for some models)
        result['amount_bin'] = pd.cut(
            result['amount'],
            bins=[0, 10, 25, 50, 100, 250, 500, 1000, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large', 'major', 'significant', 'extreme']
        )
        
        # Extract merchant name using the improved function
        from utils.text import extract_merchant_name
        result['merchant'] = result['cleaned_description'].apply(extract_merchant_name)
        
        # Canadian-specific: High-frequency merchant flag (merchants appearing 3+ times)
        merchant_counts = result['merchant'].value_counts()
        frequent_merchants = merchant_counts[merchant_counts >= 3].index.tolist()
        result['is_frequent_merchant'] = result['merchant'].apply(lambda x: 1 if x in frequent_merchants else 0)
        
        # Canadian-specific: Recurring transaction detection
        # Look for similar descriptions that occur monthly with similar amounts
        result['month_year'] = result['date'].dt.strftime('%Y-%m')
        
        # Group by merchant and month, check for recurring patterns
        merchant_monthly = result.groupby(['merchant', 'month_year']).size().reset_index(name='freq')
        recurring_merchants = merchant_monthly.groupby('merchant')['freq'].count()
        recurring_merchants = recurring_merchants[recurring_merchants >= 3].index.tolist()
        
        result['likely_recurring'] = result['merchant'].apply(lambda x: 1 if x in recurring_merchants else 0)
        
        # Text-based features: presence of keywords
        grocery_keywords = ['grocer', 'food', 'farm', 'market', 'supermarket', 'sobeys', 'loblaws', 
                        'metro', 'savemart', 'foodbasic', 'nofrills', 'bulk barn']
        dining_keywords = ['restaur', 'cafe', 'diner', 'grill', 'pizza', 'sushi', 'bistro', 
                        'pub', 'bar', 'tim horton', 'starbucks', 'mcdo', 'subway', 'swiss chalet', 'scores']
        retail_keywords = ['store', 'shop', 'mart', 'costco', 'walmart', 'canadian tire', 'ikea', 
                        'amazon', 'ebay', 'hudson bay', 'winners', 'dollarama']
        utility_keywords = ['hydro', 'electric', 'gas', 'water', 'internet', 'phone', 'mobile', 
                        'bell', 'rogers', 'telus', 'fido', 'enbridge']
        transport_keywords = ['transit', 'bus', 'train', 'subway', 'taxi', 'uber', 'lyft', 'presto', 'go train', 'via rail']
        
        # Check for keyword presence in descriptions
        result['has_grocery_kw'] = result['cleaned_description'].apply(
            lambda x: 1 if any(kw in x for kw in grocery_keywords) else 0
        )
        result['has_dining_kw'] = result['cleaned_description'].apply(
            lambda x: 1 if any(kw in x for kw in dining_keywords) else 0
        )
        result['has_retail_kw'] = result['cleaned_description'].apply(
            lambda x: 1 if any(kw in x for kw in retail_keywords) else 0
        )
        result['has_utility_kw'] = result['cleaned_description'].apply(
            lambda x: 1 if any(kw in x for kw in utility_keywords) else 0
        )
        result['has_transport_kw'] = result['cleaned_description'].apply(
            lambda x: 1 if any(kw in x for kw in transport_keywords) else 0
        )
        
        # Clean up temporary columns
        if 'month_year' in result.columns:
            result = result.drop('month_year', axis=1)
        
        return result
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract useful features from transaction data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with additional extracted features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Clean descriptions
        result['cleaned_description'] = result['description'].apply(self.clean_description)
        
        # Convert dates to datetime
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        
        # Extract date features
        result['day_of_week'] = result['date'].dt.dayofweek
        result['month'] = result['date'].dt.month
        result['is_weekend'] = result['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Convert amount to numeric
        result['amount'] = pd.to_numeric(result['amount'], errors='coerce')
        
        # Create amount bins (useful for some models)
        result['amount_bin'] = pd.cut(
            result['amount'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
        
        # Extract merchant name (simple heuristic)
        result['merchant'] = result['cleaned_description'].apply(
            lambda x: x.split()[0] if x and len(x.split()) > 0 else 'unknown'
        )
        
        logger.info(f"Extracted features from {len(df)} transactions")
        return result
    
    def prepare_for_training(self, df: pd.DataFrame, 
                             test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training by splitting into train and test sets.
        
        Args:
            df: DataFrame with transaction data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First extract features
        processed_df = self.extract_features(df)
        
        # Split the data
        train_df, test_df = train_test_split(
            processed_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=processed_df['category'] if len(processed_df['category'].unique()) > 1 else None
        )
        
        logger.info(f"Split data into {len(train_df)} training and {len(test_df)} testing samples")
        
        return train_df, test_df
    
    def get_categories(self) -> List[str]:
        """
        Get the list of unique categories found in the data.
        
        Returns:
            List of category names
        """
        return sorted(list(self.categories))
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed transaction data to a CSV file.
        
        Args:
            df: DataFrame with processed transaction data
            output_path: Path to save the processed data
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data with {len(df)} transactions to {output_path}")

# Example usage
if __name__ == "__main__":
    # This is just for testing the module directly
    processor = TransactionProcessor()
    
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
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_data)
        temp_file = f.name
    
    try:
        # Load and process the data
        df = processor.load_data(temp_file)
        processed_df = processor.extract_features(df)
        
        # Print the results
        print(f"Loaded {len(df)} transactions")
        print(f"Found {len(processor.get_categories())} categories: {processor.get_categories()}")
        print("\nSample processed data:")
        print(processed_df.head())
        
        # Split for training
        train_df, test_df = processor.prepare_for_training(df)
        print(f"\nTraining data: {len(train_df)} rows")
        print(f"Testing data: {len(test_df)} rows")
    
    finally:
        # Clean up
        os.unlink(temp_file)
