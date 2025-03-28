import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
import csv

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of transactions to generate
NUM_TRANSACTIONS = 10000
BATCH_SIZE = 1000

# Define vendors that can appear in multiple categories
multi_category_vendors = {
    # Retail + Groceries
    'WALMART #': ['Retail', 'Groceries'],
    'COSTCO WHOLESALE #': ['Retail', 'Groceries'],
    'SUPERSTORE #': ['Retail', 'Groceries'],
    'LOBLAWS #': ['Retail', 'Groceries'],
    'WALMART SUPERCENTRE #': ['Retail', 'Groceries'],
    'TARGET #': ['Retail', 'Groceries'],
    'METRO PLUS #': ['Retail', 'Groceries'],
    'GIANT TIGER #': ['Retail', 'Groceries'],
    'FRESHCO #': ['Retail', 'Groceries'],
    'NO FRILLS #': ['Retail', 'Groceries'],
    
    # Retail + Transportation
    'CANADIAN TIRE #': ['Retail', 'Transportation', 'Miscellaneous'],
    'COSTCO GAS #': ['Retail', 'Transportation'],
    'CAA #': ['Retail', 'Transportation', 'Miscellaneous'],
    'BASS PRO SHOPS #': ['Retail', 'Transportation'],
    'WALMART TIRE & AUTO #': ['Retail', 'Transportation'],
    
    # Retail + Health
    'SHOPPERS DRUG MART #': ['Retail', 'Health', 'Groceries'],
    'LONDON DRUGS #': ['Retail', 'Health', 'Groceries'],
    'JEAN COUTU #': ['Retail', 'Health'],
    'REXALL #': ['Retail', 'Health'],
    'PHARMASAVE #': ['Retail', 'Health'],
    'LAWTONS DRUGS #': ['Retail', 'Health'],
    'FAMILIPRIX #': ['Retail', 'Health'],
    'UNIPRIX #': ['Retail', 'Health'],
    'COSTCO PHARMACY #': ['Retail', 'Health'],
    'WALMART PHARMACY #': ['Retail', 'Health'],
    'GUARDIAN PHARMACY #': ['Retail', 'Health'],
    
    # Dining + Retail
    'TIM HORTONS #': ['Dining', 'Groceries'],
    'STARBUCKS #': ['Dining', 'Retail'],
    'SECOND CUP #': ['Dining', 'Retail'],
    'COBS BREAD #': ['Dining', 'Groceries'],
    'PANERA BREAD #': ['Dining', 'Groceries'],
    'BULK BARN #': ['Retail', 'Groceries'],
    'MCDONALDS #': ['Dining', 'Groceries'],
    'A&W #': ['Dining', 'Groceries'],
    
    # Gas Stations + Groceries
    'ESSO #': ['Transportation', 'Groceries', 'Retail'],
    'PETRO-CANADA #': ['Transportation', 'Groceries', 'Retail'],
    'SHELL #': ['Transportation', 'Groceries', 'Retail'],
    'ULTRAMAR #': ['Transportation', 'Groceries', 'Retail'],
    'HUSKY #': ['Transportation', 'Groceries', 'Retail'],
    'PIONEER #': ['Transportation', 'Groceries', 'Retail'],
    'CIRCLE K #': ['Transportation', 'Groceries', 'Retail'],
    '7-ELEVEN #': ['Transportation', 'Groceries', 'Retail'],
    'COUCHE-TARD #': ['Transportation', 'Groceries', 'Retail'],
    'MAC\'S #': ['Transportation', 'Groceries', 'Retail'],
    
    # Banks + Retail
    'SCOTIABANK #': ['Banking', 'Retail'],
    'TD BANK #': ['Banking', 'Retail'],
    'RBC ROYAL BANK #': ['Banking', 'Retail'],
    'BMO #': ['Banking', 'Retail'],
    'CIBC #': ['Banking', 'Retail'],
    
    # Entertainment + Retail
    'INDIGO CHAPTERS #': ['Entertainment', 'Retail'],
    'BESTBUY #': ['Entertainment', 'Retail'],
    'THE SOURCE #': ['Entertainment', 'Retail'],
    'EB GAMES #': ['Entertainment', 'Retail'],
    'APPLE STORE #': ['Entertainment', 'Retail', 'Miscellaneous'],
    
    # Miscellaneous multi-category
    'AMAZON.CA*': ['Retail', 'Groceries', 'Entertainment', 'Miscellaneous'],
    'COSTCO.CA': ['Retail', 'Groceries', 'Miscellaneous'],
    'WALMART.CA': ['Retail', 'Groceries', 'Miscellaneous'],
    'EBAY*': ['Retail', 'Entertainment', 'Miscellaneous'],
    'ETSY*': ['Retail', 'Miscellaneous'],
    'CANADIAN TIRE GAS #': ['Transportation', 'Retail'],
    'COLLEGE/UNIVERSITY #': ['Miscellaneous', 'Dining'],
    'HOSPITAL CAFE #': ['Dining', 'Health'],
    'HOTEL STAY-': ['Miscellaneous', 'Dining'],
    'LOBLAW ONLINE': ['Groceries', 'Retail'],
    'SOBEYS ONLINE': ['Groceries', 'Retail'],
    'METRO ONLINE': ['Groceries', 'Retail'],
    'INSTACART*': ['Groceries', 'Retail', 'Miscellaneous'],
    'UBER EATS': ['Dining', 'Miscellaneous'],
    'SKIP THE DISHES': ['Dining', 'Miscellaneous'],
    'DOORDASH': ['Dining', 'Miscellaneous'],
    'GOVERNMENT OF CANADA': ['Miscellaneous', 'Income']
}

# Define categories and their respective vendors/descriptions
transaction_categories = {
    'Groceries': [
        'LOBLAWS #', 'METRO #', 'SOBEYS #', 'NO FRILLS #', 'FRESHCO #', 
        'IGA #', 'SAVE-ON-FOODS #', 'FOODLAND #', 
        'SAFEWAY #', 'FARM BOY #', 'T&T SUPERMARKET #', 'WHOLE FOODS #',
        'LONGOS #', 'FORTINOS #', 'ZEHRS #', 'PROVIGO #', 'MAXI #',
        # Multi-category vendors will be added programmatically
    ],
    
    'Dining': [
        'SECOND CUP #', 'A&W #', 'HARVEYS #',
        'MCDONALDS #', 'SUBWAY #', 'SWISS CHALET #', 'THE KEG #', 'EARLS #',
        'BOSTON PIZZA #', 'ST-HUBERT #', 'CACTUS CLUB #', 'MONTANA\'S #',
        'MOXIES #', 'JOEY #', 'JACK ASTORS #', 'MILESTONES #', 'EAST SIDE MARIO\'S #',
        'BROWNS SOCIALHOUSE #', 'KELSEY\'S #', 'CORA #', 'SCORES #'
        # Multi-category vendors will be added programmatically
    ],
    
    'Retail': [
        'THE BAY #', 
        'WINNERS #', 'HOMESENSE #', 'MARSHALLS #', 'SPORTCHEK #', 'BEST BUY #',
        'RONA #', 'HOME DEPOT #', 'STAPLES #', 'INDIGO CHAPTERS #', 'IKEA #',
        'DOLLARAMA #', 'GIANT TIGER #', 'ROOTS #', 'MARK\'S #'
        # Multi-category vendors will be added programmatically
    ],
    
    'Transportation': [
        'PETRO-CANADA #', 'ESSO #', 'SHELL #', 'ULTRAMAR #', 'HUSKY #',
        'CHEVRON #', 'PIONEER #', 'CANADIAN TIRE GAS #', 'COSTCO GAS #',
        'VIA RAIL #', 'AIR CANADA #', 'WESTJET #', 'PORTER AIRLINES #',
        'GO TRANSIT #', 'TTC #', 'OC TRANSPO #', 'STM #', 'BC FERRIES #', 
        'UBER *TRIP', 'LYFT *RIDE'
    ],
    
    'Utilities': [
        'HYDRO-QUÉBEC', 'BC HYDRO', 'ONTARIO HYDRO', 'ENMAX', 'EPCOR',
        'FORTISBC', 'ALECTRA UTILITIES', 'TORONTO HYDRO', 'ATCO GAS', 
        'ENBRIDGE GAS', 'ROGERS COMMUNICATIONS', 'BELL CANADA', 'TELUS',
        'SHAW COMMUNICATIONS', 'VIDEOTRON', 'COGECO', 'EASTLINK',
        'FREEDOM MOBILE', 'FIDO MOBILE'
    ],
    
    'Entertainment': [
        'CINEPLEX #', 'LANDMARK CINEMAS #', 'NETFLIX SUBSCRIPTION',
        'SPOTIFY SUBSCRIPTION', 'APPLE ITUNES', 'AMAZON PRIME SUBSCRIPTION',
        'DISNEY PLUS SUBSCRIPTION', 'CRAVE SUBSCRIPTION', 'APPLE TV+',
        'GOOGLE PLAY STORE', 'XBOX LIVE SUBSCRIPTION', 'PLAYSTATION NETWORK',
        'NINTENDO ESHOP', 'TSN DIRECT', 'SPORTSNET NOW'
    ],
    
    'Health': [
        'GOODLIFE FITNESS #', 'ANYTIME FITNESS #', 'YMCA #', 'FITNESS WORLD #',
        'LA FITNESS #', 'PLANET FITNESS #', 'ORANGETHEORY FITNESS #',
        'REXALL PHARMA PLUS #', 'SHOPPERS DRUG MART PHARMACY #', 'LONDON DRUGS PHARMACY #',
        'JEAN COUTU PHARMACY #', 'LIFE LABS #', 'DYNACARE #'
    ],
    
    'Banking': [
        'INTERAC E-TRANSFER SENT', 'INTERAC E-TRANSFER RECEIVED',
        'TRANSFER TO SAVINGS', 'TRANSFER TO CHEQUING', 'TRANSFER TO TFSA',
        'TRANSFER TO RRSP', 'MORTGAGE PAYMENT', 'LOAN PAYMENT', 'LINE OF CREDIT PAYMENT',
        'CREDIT CARD PAYMENT', 'OVERDRAFT PROTECTION FEE', 'NSF FEE',
        'MONTHLY ACCOUNT FEE', 'ATM WITHDRAWAL', 'ATM DEPOSIT',
        'ONLINE BILL PAYMENT', 'PRE-AUTHORIZED PAYMENT'
    ],
    
    'Income': [
        'DIRECT DEPOSIT - PAYROLL', 'CANADA CHILD BENEFIT',
        'CPP PAYMENT', 'OAS PAYMENT', 'EI PAYMENT', 'GST/HST CREDIT',
        'CERB PAYMENT', 'CRB PAYMENT', 'CRSB PAYMENT',
        'INTEREST PAYMENT', 'DIVIDEND PAYMENT', 'TAX REFUND - CRA'
    ],
    
    'Miscellaneous': [
        'AMAZON.CA*', 'PAYPAL *PURCHASE', 'ETSY.COM PURCHASE', 'DONATION-',
        'INSURANCE PREMIUM', 'PROPERTY TAX PAYMENT', 'MEMBERSHIP FEE',
        'TUITION PAYMENT', 'TICKETMASTER EVENT', 'AIRBNB STAY', 'HOTEL STAY-',
        'GROUPON PURCHASE', 'EXPEDIA BOOKING'
    ]
}

# Add location numbers and random details to vendor names
def add_details(vendor):
    if '#' in vendor:
        return vendor.replace('#', str(random.randint(100, 999)))
    return vendor

# Generate random amounts based on category
def generate_amount(category):
    if category == 'Groceries':
        return round(random.uniform(20, 200), 2)
    elif category == 'Dining':
        return round(random.uniform(5, 100), 2)
    elif category == 'Retail':
        return round(random.uniform(10, 500), 2)
    elif category == 'Transportation':
        return round(random.uniform(20, 150), 2)
    elif category == 'Utilities':
        return round(random.uniform(50, 300), 2)
    elif category == 'Entertainment':
        return round(random.uniform(10, 80), 2)
    elif category == 'Health':
        return round(random.uniform(20, 200), 2)
    elif category == 'Banking':
        return round(random.uniform(50, 2000), 2)
    elif category == 'Income':
        return round(random.uniform(500, 5000), 2)
    else:  # Miscellaneous
        return round(random.uniform(10, 300), 2)

# Generate random dates spanning the last year
def generate_date():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    
    random_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_days)
    
    return random_date.strftime('%Y-%m-%d')

# Add some variation to descriptions
def add_variation(description, category):
    variations = {
        'Groceries': ['PURCHASE', 'GROCERIES', ''],
        'Dining': ['PURCHASE', 'MEAL', 'RESTAURANT', ''],
        'Retail': ['PURCHASE', 'SHOPPING', ''],
        'Transportation': ['FUEL', 'GAS', 'TICKET', 'FARE', ''],
        'Utilities': ['BILL PAYMENT', 'MONTHLY BILL', 'SERVICE PAYMENT', ''],
        'Entertainment': ['SUBSCRIPTION', 'PAYMENT', 'ENTERTAINMENT', ''],
        'Health': ['SERVICE', 'MEMBERSHIP', 'PURCHASE', ''],
        'Banking': ['', 'FEE', 'TRANSACTION'],
        'Income': ['DEPOSIT', 'PAYMENT', ''],
        'Miscellaneous': ['PURCHASE', 'PAYMENT', 'TRANSACTION', '']
    }
    
    # 70% chance to add a variation
    if random.random() < 0.7:
        variation = random.choice(variations.get(category, ['']))
        if variation and not description.endswith(variation):
            return f"{description} {variation}"
    
    return description

# Generate transactions
def generate_transactions(num_transactions=NUM_TRANSACTIONS):
    transactions = []
    
    categories = list(transaction_categories.keys())
    category_weights = [
        0.18,  # Groceries
        0.15,  # Dining
        0.12,  # Retail
        0.1,   # Transportation
        0.08,  # Utilities
        0.08,  # Entertainment
        0.07,  # Health
        0.1,   # Banking
        0.07,  # Income
        0.05   # Miscellaneous
    ]
    
    # Create a mapping of vendors to possible categories
    vendor_to_categories = {}
    for category, vendors in transaction_categories.items():
        for vendor in vendors:
            if vendor not in vendor_to_categories:
                vendor_to_categories[vendor] = []
            vendor_to_categories[vendor].append(category)
    
    for _ in range(num_transactions):
        # Select category based on weights
        category = random.choices(categories, weights=category_weights, k=1)[0]
        
        # Select vendor/description from the category
        base_description = random.choice(transaction_categories[category])
        description = add_details(base_description)
        
        # For multi-category vendors, we may want to randomly assign a more specific category
        if base_description in multi_category_vendors:
            possible_categories = multi_category_vendors[base_description]
            # 70% chance to reassign the category if this vendor belongs to multiple categories
            if random.random() < 0.7 and category in possible_categories:
                # Keep the same description but potentially change the category
                # to one of the vendor's possible categories
                category = random.choice(possible_categories)
        
        description = add_variation(description, category)
        
        # Generate other transaction details
        amount = generate_amount(category)
        date = generate_date()
        
        transactions.append({
            'date': date,
            'description': description,
            'amount': amount,
            'category': category
        })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(transactions)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Add special cases for common financial activities
def add_special_cases(df):
    # Generate some special common transactions
    special_cases = [
        {'description': 'TRANSFER TO SAVINGS', 'category': 'Banking', 'amount': round(random.uniform(100, 2000), 2)},
        {'description': 'TRANSFER TO TFSA', 'category': 'Banking', 'amount': round(random.uniform(500, 6000), 2)},
        {'description': 'TRANSFER TO RRSP', 'category': 'Banking', 'amount': round(random.uniform(500, 6000), 2)},
        {'description': 'TRANSFER TO CHECKING', 'category': 'Banking', 'amount': round(random.uniform(100, 2000), 2)},
        {'description': 'TRANSFER TO INVESTMENT ACCOUNT', 'category': 'Banking', 'amount': round(random.uniform(1000, 10000), 2)},
        {'description': 'MORTGAGE PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(800, 3000), 2)},
        {'description': 'DIRECT DEPOSIT - PAYROLL ABC COMPANY', 'category': 'Income', 'amount': round(random.uniform(1500, 5000), 2)},
        {'description': 'DIRECT DEPOSIT - PAYROLL XYZ CORP', 'category': 'Income', 'amount': round(random.uniform(1500, 5000), 2)},
        {'description': 'DIRECT DEPOSIT - PAYROLL GOVERNMENT OF CANADA', 'category': 'Income', 'amount': round(random.uniform(1800, 6000), 2)},
        {'description': 'DIRECT DEPOSIT - PAYROLL CITY OF TORONTO', 'category': 'Income', 'amount': round(random.uniform(1800, 5500), 2)},
        {'description': 'DIRECT DEPOSIT - PAYROLL PROVINCE OF BC', 'category': 'Income', 'amount': round(random.uniform(1800, 5500), 2)},
        {'description': 'INTERAC E-TRANSFER FROM JOHN SMITH', 'category': 'Income', 'amount': round(random.uniform(20, 200), 2)},
        {'description': 'INTERAC E-TRANSFER FROM MARY WILSON', 'category': 'Income', 'amount': round(random.uniform(20, 500), 2)},
        {'description': 'INTERAC E-TRANSFER TO JANE DOE', 'category': 'Banking', 'amount': round(random.uniform(20, 200), 2)},
        {'description': 'INTERAC E-TRANSFER TO ROBERT JOHNSON', 'category': 'Banking', 'amount': round(random.uniform(20, 300), 2)},
        {'description': 'ATM WITHDRAWAL TD BANK', 'category': 'Banking', 'amount': round(random.choice([20, 40, 60, 80, 100]), 2)},
        {'description': 'ATM WITHDRAWAL RBC', 'category': 'Banking', 'amount': round(random.choice([20, 40, 60, 80, 100]), 2)},
        {'description': 'ATM WITHDRAWAL SCOTIABANK', 'category': 'Banking', 'amount': round(random.choice([20, 40, 60, 80, 100]), 2)},
        {'description': 'ATM WITHDRAWAL BMO', 'category': 'Banking', 'amount': round(random.choice([20, 40, 60, 80, 100]), 2)},
        {'description': 'ATM WITHDRAWAL CIBC', 'category': 'Banking', 'amount': round(random.choice([20, 40, 60, 80, 100]), 2)},
        {'description': 'CREDIT CARD PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(100, 1500), 2)},
        {'description': 'VISA PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(100, 2000), 2)},
        {'description': 'MASTERCARD PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(100, 2000), 2)},
        {'description': 'AMEX PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(200, 3000), 2)},
        {'description': 'LINE OF CREDIT PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(200, 1000), 2)},
        {'description': 'LOAN PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(300, 1200), 2)},
        {'description': 'CAR LOAN PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(300, 800), 2)},
        {'description': 'STUDENT LOAN PAYMENT', 'category': 'Banking', 'amount': round(random.uniform(200, 600), 2)},
        {'description': 'ONLINE BILL PAYMENT - PROPERTY TAX', 'category': 'Miscellaneous', 'amount': round(random.uniform(1000, 5000), 2)},
        {'description': 'ONLINE BILL PAYMENT - INSURANCE', 'category': 'Miscellaneous', 'amount': round(random.uniform(100, 500), 2)},
        {'description': 'ONLINE BILL PAYMENT - CAR INSURANCE', 'category': 'Miscellaneous', 'amount': round(random.uniform(100, 300), 2)},
        {'description': 'ONLINE BILL PAYMENT - HOME INSURANCE', 'category': 'Miscellaneous', 'amount': round(random.uniform(100, 400), 2)},
        {'description': 'ONLINE BILL PAYMENT - LIFE INSURANCE', 'category': 'Miscellaneous', 'amount': round(random.uniform(50, 300), 2)},
        {'description': 'CPP PAYMENT', 'category': 'Income', 'amount': round(random.uniform(500, 1200), 2)},
        {'description': 'OAS PAYMENT', 'category': 'Income', 'amount': round(random.uniform(500, 700), 2)},
        {'description': 'EI PAYMENT', 'category': 'Income', 'amount': round(random.uniform(800, 1800), 2)},
        {'description': 'CANADA CHILD BENEFIT', 'category': 'Income', 'amount': round(random.uniform(300, 1000), 2)},
        {'description': 'GST/HST CREDIT', 'category': 'Income', 'amount': round(random.uniform(100, 400), 2)},
        {'description': 'ONTARIO TRILLIUM BENEFIT', 'category': 'Income', 'amount': round(random.uniform(100, 500), 2)},
        {'description': 'BC CLIMATE ACTION TAX CREDIT', 'category': 'Income', 'amount': round(random.uniform(100, 300), 2)},
        {'description': 'ALBERTA CHILD BENEFIT', 'category': 'Income', 'amount': round(random.uniform(200, 600), 2)},
        {'description': 'INTEREST EARNED', 'category': 'Income', 'amount': round(random.uniform(0.1, 50), 2)},
        {'description': 'MONTHLY ACCOUNT FEE', 'category': 'Banking', 'amount': round(random.uniform(4, 30), 2)},
        {'description': 'ANNUAL ACCOUNT FEE', 'category': 'Banking', 'amount': round(random.uniform(50, 150), 2)},
        {'description': 'OVERDRAFT FEE', 'category': 'Banking', 'amount': round(random.uniform(30, 45), 2)},
        {'description': 'NSF FEE', 'category': 'Banking', 'amount': round(random.uniform(40, 50), 2)},
        {'description': 'DIVIDEND PAYMENT - TD BANK', 'category': 'Income', 'amount': round(random.uniform(10, 500), 2)},
        {'description': 'DIVIDEND PAYMENT - RBC', 'category': 'Income', 'amount': round(random.uniform(10, 500), 2)},
        {'description': 'DIVIDEND PAYMENT - BCE', 'category': 'Income', 'amount': round(random.uniform(10, 400), 2)},
        {'description': 'DIVIDEND PAYMENT - ENBRIDGE', 'category': 'Income', 'amount': round(random.uniform(10, 300), 2)},
        {'description': 'DIVIDEND PAYMENT - TELUS', 'category': 'Income', 'amount': round(random.uniform(10, 250), 2)}
    ]
    
    # Add dates to special cases
    for transaction in special_cases:
        transaction['date'] = generate_date()
    
    # Convert to DataFrame and concat with main df if provided
    special_df = pd.DataFrame(special_cases)
    
    if len(df) > 0:
        result_df = pd.concat([df, special_df], ignore_index=True)
        # Sort by date
        result_df = result_df.sort_values(by='date').reset_index(drop=True)
        return result_df
    else:
        # If no dataframe was provided, just return the special cases
        return special_df.sort_values(by='date').reset_index(drop=True)

# Add multi-category vendors to their respective categories
def add_multi_category_vendors():
    # First add each multi-category vendor to their respective categories
    for vendor, categories in multi_category_vendors.items():
        for category in categories:
            if category in transaction_categories and vendor not in transaction_categories[category]:
                transaction_categories[category].append(vendor)

# Main function to generate the dataset
def generate_dataset(num_transactions=NUM_TRANSACTIONS, output_file='canadian_transaction_data.csv', batch_size=BATCH_SIZE):
    print(f"Generating dataset with {num_transactions} transactions...")
    
    # Add multi-category vendors to their respective categories
    add_multi_category_vendors()
    
    # Check if we need to batch process
    if num_transactions <= batch_size:
        # Small enough to generate in one go
        df = generate_transactions(num_transactions)
        df = add_special_cases(df)
    else:
        # Need to batch process
        print(f"Processing in batches of {batch_size} transactions...")
        dfs = []
        remaining = num_transactions
        batch_num = 1
        
        while remaining > 0:
            batch_size_curr = min(batch_size, remaining)
            print(f"Generating batch {batch_num} ({batch_size_curr} transactions)...")
            batch_df = generate_transactions(batch_size_curr)
            dfs.append(batch_df)
            remaining -= batch_size_curr
            batch_num += 1
        
        # Combine all batches
        df = pd.concat(dfs, ignore_index=True)
        
        # Add special cases, making sure we don't exceed requested count
        special_df = add_special_cases(pd.DataFrame())
        df = pd.concat([df, special_df], ignore_index=True)
        if len(df) > num_transactions:
            df = df.sample(n=num_transactions)
        
        # Final shuffle
        df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Dataset generated successfully: {output_file}")
    print(f"Total transactions: {len(df)}")
    
    # Display category distribution
    category_counts = df['category'].value_counts()
    print("\nCategory distribution:")
    for category, count in category_counts.items():
        print(f"{category}: {count} ({count/len(df)*100:.1f}%)")
    
    # Display multi-category vendor distribution (top 20)
    multi_vendor_counts = {}
    for vendor in multi_category_vendors.keys():
        vendor_base = vendor.replace('#', '')
        count = df['description'].str.contains(vendor_base).sum()
        multi_vendor_counts[vendor] = count
    
    print("\nMulti-category vendor distribution (top 20):")
    for vendor, count in sorted(multi_vendor_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{vendor}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate synthetic Canadian transaction data")
    parser.add_argument("--num-transactions", type=int, default=NUM_TRANSACTIONS,
                        help=f"Number of transactions to generate (default: {NUM_TRANSACTIONS})")
    parser.add_argument("--output", type=str, default="canadian_transaction_data.csv",
                        help="Output CSV file name (default: canadian_transaction_data.csv)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for processing large datasets (default: {BATCH_SIZE})")
    
    args = parser.parse_args()
    
    # Generate the dataset with the specified parameters
    generate_dataset(num_transactions=args.num_transactions, 
                    output_file=args.output, 
                    batch_size=args.batch_size)