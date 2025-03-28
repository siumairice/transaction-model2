# main.py
"""
Main entry point for the transaction categorization system.
Provides a command-line interface for processing data, training models,
and making predictions.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
from typing import List, Dict, Optional, Union, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_arg_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Transaction Categorization System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process transaction data')
    process_parser.add_argument(
        '--input', '-i', required=True, help='Input CSV file with transaction data'
    )
    process_parser.add_argument(
        '--output', '-o', required=True, help='Output path for processed data'
    )
    process_parser.add_argument(
        '--split', '-s', action='store_true', help='Split data into train/test sets'
    )
    process_parser.add_argument(
        '--test-size', '-t', type=float, default=0.2, help='Proportion of data for testing'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a categorization model')
    train_parser.add_argument(
        '--data', '-d', required=True, help='CSV file with processed transaction data'
    )
    train_parser.add_argument(
        '--model-dir', '-m', required=True, help='Directory to save model'
    )
    train_parser.add_argument(
        '--epochs', '-e', type=int, default=4, help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size', '-b', type=int, default=16, help='Training batch size'
    )
    train_parser.add_argument(
        '--learning-rate', '-l', type=float, default=2e-5, help='Learning rate'
    )
    train_parser.add_argument(
        '--include-amount', '-a', action='store_true', help='Include transaction amount as feature'
    )
    train_parser.add_argument(
        '--include-dates', '-dt', action='store_true', help='Include date features'
    )
    train_parser.add_argument(
        '--test-split', '-ts', type=float, default=0.2, help='Proportion of data for testing'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict categories for transactions')
    predict_parser.add_argument(
        '--model-dir', '-m', required=True, help='Directory with trained model'
    )
    predict_parser.add_argument(
        '--data', '-d', required=True, help='CSV file with transactions to categorize'
    )
    predict_parser.add_argument(
        '--output', '-o', required=True, help='Output file for predictions'
    )
    predict_parser.add_argument(
        '--description-col', '-dc', default='description', help='Column name for transaction descriptions'
    )
    predict_parser.add_argument(
        '--amount-col', '-ac', default='amount', help='Column name for transaction amounts'
    )
    predict_parser.add_argument(
        '--date-col', '-dtc', default='date', help='Column name for transaction dates'
    )
    predict_parser.add_argument(
        '--batch-size', '-b', type=int, default=32, help='Batch size for prediction'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument(
        '--model-dir', '-m', required=True, help='Directory with trained model'
    )
    evaluate_parser.add_argument(
        '--data', '-d', required=True, help='CSV file with transactions and true categories'
    )
    evaluate_parser.add_argument(
        '--output-dir', '-o', default=None, help='Directory to save evaluation results'
    )
    evaluate_parser.add_argument(
        '--description-col', '-dc', default='description', help='Column name for transaction descriptions'
    )
    evaluate_parser.add_argument(
        '--category-col', '-cc', default='category', help='Column name for true categories'
    )
    evaluate_parser.add_argument(
        '--amount-col', '-ac', default='amount', help='Column name for transaction amounts'
    )
    evaluate_parser.add_argument(
        '--date-col', '-dtc', default='date', help='Column name for transaction dates'
    )
    
    return parser

def process_command(args: argparse.Namespace) -> None:
    """
    Handle the 'process' command to prepare data.
    
    Args:
        args: Command-line arguments
    """
    from data.processor import TransactionProcessor
    
    logger.info(f"Processing transaction data from {args.input}")
    
    # Initialize the processor
    processor = TransactionProcessor(
        remove_numbers=True,
        remove_special_chars=True,
        lowercase=True
    )
    
    try:
        # Load data
        df = processor.load_data(args.input)
        logger.info(f"Loaded {len(df)} transactions with {len(processor.get_categories())} categories")
        
        # Process data
        processed_df = processor.extract_features(df)
        
        if args.split:
            # Split into train/test and save separately
            train_df, test_df = processor.prepare_for_training(
                processed_df, test_size=args.test_size
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Save train data
            train_output = os.path.join(
                os.path.dirname(args.output),
                f"{os.path.splitext(os.path.basename(args.output))[0]}_train.csv"
            )
            processor.save_processed_data(train_df, train_output)
            
            # Save test data
            test_output = os.path.join(
                os.path.dirname(args.output),
                f"{os.path.splitext(os.path.basename(args.output))[0]}_test.csv"
            )
            processor.save_processed_data(test_df, test_output)
            
            logger.info(f"Saved train data ({len(train_df)} rows) to {train_output}")
            logger.info(f"Saved test data ({len(test_df)} rows) to {test_output}")
        else:
            # Save all processed data to a single file
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            processor.save_processed_data(processed_df, args.output)
            logger.info(f"Saved processed data ({len(processed_df)} rows) to {args.output}")
        
        # Save category information
        categories_output = os.path.join(
            os.path.dirname(args.output),
            "categories.txt"
        )
        with open(categories_output, 'w') as f:
            for category in processor.get_categories():
                f.write(f"{category}\n")
        logger.info(f"Saved {len(processor.get_categories())} categories to {categories_output}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        sys.exit(1)

def train_command(args: argparse.Namespace) -> None:
    """
    Handle the 'train' command to train a model.
    
    Args:
        args: Command-line arguments
    """
    from data.processor import TransactionProcessor
    from data.dataset import create_data_loaders
    from models.classifier import create_transaction_model
    from training.trainer import train_transaction_model
    from transformers import BertTokenizer
    
    logger.info(f"Training model using data from {args.data}")
    
    try:
        # Load data
        processor = TransactionProcessor()
        df = processor.load_data(args.data)
        
        # Get categories
        categories = processor.get_categories()
        logger.info(f"Found {len(categories)} categories: {categories}")
        
        # Split data if not already split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=args.test_split, random_state=42,
            stratify=df['category'] if len(df['category'].unique()) > 1 else None
        )
        
        logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation samples")
        
        # Create category mappings
        category_to_id = {cat: idx for idx, cat in enumerate(categories)}
        id_to_category = {idx: cat for idx, cat in enumerate(categories)}
        
        # Add label column
        train_df['label'] = train_df['category'].map(category_to_id)
        val_df['label'] = val_df['category'].map(category_to_id)
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create dataloaders
        train_loader, val_loader = create_data_loaders(
            train_df=train_df,
            test_df=val_df,
            tokenizer=tokenizer,
            categories=categories,
            batch_size=args.batch_size,
            max_length=128
        )
        
        # Create output directory
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Train model
        model, history = train_transaction_model(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            id_to_category=id_to_category,
            output_dir=args.model_dir,
            num_categories=len(categories),
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            include_amount=args.include_amount,
            include_date_features=args.include_dates
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(args.model_dir)
        
        # Save category mappings
        with open(os.path.join(args.model_dir, "category_mapping.json"), 'w') as f:
            json.dump({str(k): v for k, v in id_to_category.items()}, f)
        
        logger.info(f"Model training completed. Model saved to {args.model_dir}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        sys.exit(1)

def predict_command(args: argparse.Namespace) -> None:
    """
    Handle the 'predict' command to make predictions.
    
    Args:
        args: Command-line arguments
    """
    from inference.predictor import TransactionPredictor
    
    logger.info(f"Loading model from {args.model_dir}")
    logger.info(f"Making predictions for transactions in {args.data}")
    
    try:
        # Initialize predictor
        predictor = TransactionPredictor(
            model_dir=args.model_dir,
            batch_size=args.batch_size
        )
        
        # Predict
        predictor.predict_csv(
            input_file=args.data,
            output_file=args.output,
            description_col=args.description_col,
            amount_col=args.amount_col,
            date_col=args.date_col
        )
        
        logger.info(f"Predictions saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        sys.exit(1)

def evaluate_command(args: argparse.Namespace) -> None:
    """
    Handle the 'evaluate' command to evaluate model performance.
    
    Args:
        args: Command-line arguments
    """
    from inference.predictor import TransactionPredictor
    
    logger.info(f"Loading model from {args.model_dir}")
    logger.info(f"Evaluating on data from {args.data}")
    
    try:
        # Load data
        df = pd.read_csv(args.data)
        
        # Initialize predictor
        predictor = TransactionPredictor(
            model_dir=args.model_dir
        )
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Evaluate
        metrics = predictor.evaluate(
            df=df,
            description_col=args.description_col,
            true_category_col=args.category_col,
            amount_col=args.amount_col,
            date_col=args.date_col,
            output_dir=args.output_dir
        )
        
        # Print metrics
        logger.info("Evaluation results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        logger.info(f"Evaluation completed")
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
        sys.exit(1)

def main() -> None:
    """
    Main entry point for the application.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    if args.command == 'process':
        process_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()