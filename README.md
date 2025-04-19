# Transaction Categorization CLI

# Initialize Virtual Environment
```
python3 -m venv modelvenv && 
source modelvenv/bin/activate && 
pip install -r requirements.txt && cd src
```

## Usage

## Generate Mock Data

| Amounts less than 50 or so will be rounded up to 50.

```bash
python data/dataset_gen.py --num-transactions 100
```

## Training

| Recommended batch-size of 8, cleanup is to delete any synthetic data that was generated, visualize is for verbose output

```bash
python test.py --epochs 3 --batch-size 8 --cleanup --visualize
```

## Predicting

```bash
# Create a sample test file
cat > sample_transactions.csv << EOL   
date,description,amount
2025-03-01,AMAZON PRIME MEMBERSHIP,12.99
2025-03-02,STARBUCKS COFFEE,5.45
2025-03-03,CHEVRON GAS STATION,45.67
2025-03-04,NETFLIX SUBSCRIPTION,14.99
2025-03-05,TRADER JOE'S GROCERIES,87.32
EOL
```

```
# Run prediction
python main.py predict \
  --model-dir ./test_model/best_model \
  --data sample_transactions.csv \
  --output categorized_sample.csv \
  --threshold 0.01
```


