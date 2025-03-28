import subprocess
import os
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

def parse_args():
    """Parse command line arguments for the test script."""
    parser = argparse.ArgumentParser(description="Test the transaction classification model")
    parser.add_argument("--num-transactions", type=int, default=500, 
                        help="Number of transactions to generate if data doesn't exist")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--cleanup", action="store_true", 
                        help="Clean up temporary files after testing")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Create additional visualizations")
    return parser.parse_args()

def visualize_results(predictions_file, test_file, output_dir):
    """Create additional visualizations of the results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions and actual data
    try:
        predictions = pd.read_csv(predictions_file)
        test_data = pd.read_csv(test_file)
        
        # Merge data
        merged_data = pd.merge(
            predictions, 
            test_data[['description', 'category']], 
            on='description', 
            suffixes=('_pred', '_true')
        )
        
        # Create confusion matrix visualization
        plt.figure(figsize=(12, 10))
        
        # Count occurrences of each true/predicted category pair
        confusion = merged_data.groupby(['category_true', 'category_pred']).size().unstack(fill_value=0)
        
        # Plot heatmap
        import seaborn as sns
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_detailed.png'))
        
        # Create category distribution comparison
        plt.figure(figsize=(12, 6))
        
        # Count true and predicted categories
        true_counts = merged_data['category_true'].value_counts()
        pred_counts = merged_data['category_pred'].value_counts()
        
        # Combine and fill missing values with 0
        all_categories = sorted(set(true_counts.index) | set(pred_counts.index))
        true_counts = true_counts.reindex(all_categories, fill_value=0)
        pred_counts = pred_counts.reindex(all_categories, fill_value=0)
        
        # Plot
        x = range(len(all_categories))
        width = 0.35
        plt.bar([i - width/2 for i in x], true_counts, width, label='True')
        plt.bar([i + width/2 for i in x], pred_counts, width, label='Predicted')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.title('True vs Predicted Category Distribution')
        plt.xticks(x, all_categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
        
        # Create confidence distribution
        if 'confidence' in predictions.columns:
            plt.figure(figsize=(10, 6))
            
            # Create correct/incorrect column
            merged_data['correct'] = merged_data['category_pred'] == merged_data['category_true']
            
            # Plot confidence distributions
            sns.histplot(data=merged_data, x='confidence', hue='correct', bins=20, 
                         element='step', common_norm=False, stat='density')
            plt.title('Confidence Distribution for Correct and Incorrect Predictions')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        
        # Save summary statistics
        summary = {
            'accuracy': (merged_data['category_pred'] == merged_data['category_true']).mean(),
            'num_samples': len(merged_data),
            'categories': {
                'true_distribution': true_counts.to_dict(),
                'predicted_distribution': pred_counts.to_dict()
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_dir, 'test_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Visualizations saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return False

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Define consistent file paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(CURRENT_DIR, "canadian_transaction_data.csv")
    PROCESSED_BASE = os.path.join(CURRENT_DIR, "processed_data.csv")  # Base name only
    PROCESSED_TRAIN = os.path.join(CURRENT_DIR, "processed_data_train.csv")
    PROCESSED_TEST = os.path.join(CURRENT_DIR, "processed_data_test.csv")
    MODEL_DIR = os.path.join(CURRENT_DIR, "test_model")
    BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")  # Path to the best model
    PREDICTIONS_FILE = os.path.join(CURRENT_DIR, "predictions.csv")
    RESULTS_DIR = os.path.join(CURRENT_DIR, "test_results")
    
    # Generate data if needed
    if not os.path.exists(DATA_FILE):
        print("Generating synthetic data...")
        subprocess.run([
            "python", 
            os.path.join(CURRENT_DIR, "data/dataset_gen.py"), 
            "--num-transactions", 
            str(args.num_transactions)
        ])
    
    # Process data with split flag
    print("Processing data...")
    subprocess.run([
        "python", "main.py", "process", 
        "--input", DATA_FILE, 
        "--output", PROCESSED_BASE, 
        "--split"
    ])
    
    # Verify the files exist before continuing
    if not os.path.exists(PROCESSED_TRAIN) or not os.path.exists(PROCESSED_TEST):
        print(f"Error: Expected files {PROCESSED_TRAIN} and {PROCESSED_TEST} not found.")
        print("Files in current directory:", os.listdir(CURRENT_DIR))
        exit(1)
    
    # Train model
    print("Training model...")
    subprocess.run([
        "python", "main.py", "train", 
        "--data", PROCESSED_TRAIN, 
        "--model-dir", MODEL_DIR, 
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--include-amount",
        "--include-dates"
    ])
    
    # Make predictions
    print("Making predictions...")
    subprocess.run([
        "python", "main.py", "predict", 
        "--model-dir", BEST_MODEL_DIR, 
        "--data", PROCESSED_TEST, 
        "--output", PREDICTIONS_FILE
    ])
    
    # Evaluate
    print("Evaluating model...")
    subprocess.run([
        "python", "main.py", "evaluate", 
        "--model-dir", BEST_MODEL_DIR, 
        "--data", PROCESSED_TEST,
        "--output-dir", RESULTS_DIR
    ])
    
    # Create additional visualizations
    if args.visualize:
        print("Creating additional visualizations...")
        visualize_results(PREDICTIONS_FILE, PROCESSED_TEST, RESULTS_DIR)
    
    # Cleanup temporary files if requested
    if args.cleanup:
        print("Cleaning up temporary files...")
        temp_files = [
            PROCESSED_BASE,
            PROCESSED_TRAIN,
            PROCESSED_TEST,
            PREDICTIONS_FILE
        ]
        
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        
        # Don't remove the model by default as it might be useful to keep
        # Uncomment the following lines if you want to remove the model too
        # if os.path.exists(MODEL_DIR):
        #     shutil.rmtree(MODEL_DIR)
        #     print(f"Removed {MODEL_DIR}")
    
    print("Quick test completed!")

if __name__ == "__main__":
    main()