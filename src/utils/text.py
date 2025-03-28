# utils/text.py
"""
Text processing utilities specialized for financial transaction descriptions.
"""

import re
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Common merchant words to standardize
MERCHANT_ALIASES = {
    # Restaurants
    'mcd': 'mcdonalds',
    'mcdo': 'mcdonalds',
    'kfc': 'kfc',
    'pizzahut': 'pizza hut',
    'taco': 'taco bell',
    'subway': 'subway',
    
    # Retail
    'wlmrt': 'walmart',
    'wal mart': 'walmart',
    'walm': 'walmart',
    'costco': 'costco',
    'amzn': 'amazon',
    'amz': 'amazon',
    'amazon.com': 'amazon',
    'amzn mktp': 'amazon marketplace',
    'ebay': 'ebay',
    
    # Utilities & Services
    'att': 'at&t',
    'atnt': 'at&t',
    'vzw': 'verizon',
    'comcast': 'comcast',
    'netlflix': 'netflix',
    'nflx': 'netflix',
    'sptfy': 'spotify',
    
    # Banking
    'sq *': 'square',
    'paypal *': 'paypal',
    'venmo': 'venmo',
    'zelle': 'zelle',
    'ach': 'ach transfer',
    'autopay': 'automatic payment',
}

# Words commonly found in transaction descriptions that don't add meaning
TRANSACTION_NOISE_WORDS = {
    'payment', 'purchase', 'pos', 'debit', 'credit', 'online', 
    'transaction', 'web', 'ach', 'transfer', 'withdrawal', 'deposit',
    'check', 'fee', 'service', 'charge', 'bill', 'ref', 'authorization',
    'electronic', 'mobile', 'banking', 'bank', 'autopay', 'automatic',
    'pending', 'hold', 'preauthorized', 'preauth', 'temp', 'temporary',
    'international', 'domestic', 'recurring', 'subscription', 'direct',
    'card', 'ending', 'beginning', 'terminal', 'account', 'acct',
    'teller', 'atm', 'branch'
}

# Common transaction codes that appear in descriptions
TRANSACTION_CODES = {
    r'\b\d{4}\b': '',  # 4-digit numbers (often partial card numbers) 
    r'\b\d{5,6}\b': '',  # Terminal IDs
    r'#\d+': '',  # Reference numbers
    r'\b[A-Z0-9]{8,}\b': '',  # Transaction IDs
    r'\d{2}/\d{2}': '',  # Dates in MM/DD format
}
# Improved merchant extraction function for text.py

def extract_merchant_name(description: str) -> str:
    """
    Extract the primary merchant name from a transaction description.
    Uses heuristics to identify the most likely merchant.
    
    Args:
        description: Transaction description
        
    Returns:
        Extracted merchant name
    """
    # Clean the description first
    clean_desc = clean_transaction_text(description)
    
    # Split into words
    words = clean_desc.split()
    
    # If we have no words after cleaning, return unknown
    if not words:
        return "unknown"
    
    # Check if we have a standardized merchant name from our aliases
    for standard_name in set(MERCHANT_ALIASES.values()):
        if standard_name in clean_desc:
            return standard_name
    
    # Common Canadian merchant detection
    # Names that commonly appear with specific patterns
    canadian_merchants = {
        'costco': ['costco', 'wholesale'],
        'tim hortons': ['tim horton', 'timhorton'],
        'walmart': ['walmart', 'wal-mart', 'wal mart'],
        'canadian tire': ['cdn tire', 'canadian tire'],
        'shoppers drug mart': ['shoppers', 'shoppersdrugmart'],
        'loblaws': ['loblaws', 'superstore'],
        'sobeys': ['sobeys'],
        'metro': ['metro grocer'],
        'dollarama': ['dollarama'],
        'lcbo': ['lcbo'],
        'starbucks': ['starbucks'],
        'mcdonald': ['mcdonald', 'mcdo'],
        'second cup': ['second cup'],
        'swiss chalet': ['swiss chalet'],
        'bulk barn': ['bulk barn'],
        'scores': ['scores'],
        'boston pizza': ['boston pizza'],
        'the bay': ['hudson bay', 'the bay'],
        'roots': ['roots canada'],
        'home depot': ['home depot'],
        'no frills': ['no frills', 'nofrills'],
        'the keg': ['the keg'],
        'montana': ['montana'],
        'petro canada': ['petro-can', 'petro canada'],
        'esso': ['esso'],
        'shell': ['shell'],
        'cineplex': ['cineplex'],
        'chapters': ['chapters', 'indigo'],
        'winners': ['winners'],
        'sport chek': ['sport chek'],
        'ikea': ['ikea'],
        'ebay': ['ebay'],
        'amazon': ['amzn', 'amazon']
    }
    
    # Try to match against known Canadian merchants
    for merchant, patterns in canadian_merchants.items():
        for pattern in patterns:
            if pattern in clean_desc:
                return merchant
    
    # If merchant isn't recognized, apply some heuristics
    
    # 1. Compound merchant names (take the first two words if they're probably part of the name)
    if len(words) >= 2:
        # Words that suggest the next word is still part of the merchant name
        connectors = ['shop', 'store', 'restaurant', 'cafe', 'coffee', 'diner', 'pizza', 'grill', 
                      'bistro', 'bar', 'pub', 'market', 'supermarket', 'gas', 'station']
        
        if words[0].lower() in connectors or len(words[0]) <= 2:
            # First word is likely a descriptor, include the second word
            return ' '.join(words[:2])
    
    # 2. For single-word merchants
    if words and len(words[0]) >= 3:  # Avoid very short words
        return words[0]
    elif len(words) >= 2:
        return ' '.join(words[:2])
    
    # Fallback to the first part of the description
    merchant = description.split()[0] if description else "unknown"
    return merchant.lower()

def standardize_merchant(description: str) -> str:
    """
    Standardize merchant names in transaction descriptions.
    
    Args:
        description: Transaction description
        
    Returns:
        Standardized description
    """
    # Convert to lowercase for consistency
    text = description.lower()
    
    # Replace merchant aliases
    for alias, standard_name in MERCHANT_ALIASES.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(alias) + r'\b'
        text = re.sub(pattern, standard_name, text)
    
    return text

def clean_transaction_text(description: str, 
                          remove_noise: bool = True,
                          remove_numbers: bool = True,
                          remove_special_chars: bool = True) -> str:
    """
    Clean transaction description text by removing noise and standardizing format.
    
    Args:
        description: Transaction description
        remove_noise: Whether to remove common noise words
        remove_numbers: Whether to remove numbers
        remove_special_chars: Whether to remove special characters
        
    Returns:
        Cleaned description
    """
    if not description:
        return ""
    
    # Convert to lowercase
    text = description.lower()
    
    # Standardize merchant names
    text = standardize_merchant(text)
    
    # Remove transaction codes
    for pattern, replacement in TRANSACTION_CODES.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle special patterns in transactions
    # Replace common patterns like * or # that separate parts of descriptions
    text = re.sub(r'\s*[*#]\s*', ' ', text)
    
    # Remove special characters if specified
    if remove_special_chars:
        # Keep apostrophes and hyphens as they may be part of names
        text = re.sub(r'[^\w\s\'-]', ' ', text)
    
    # Remove numbers if specified
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common noise words if specified
    if remove_noise:
        words = text.split()
        filtered_words = [word for word in words if word not in TRANSACTION_NOISE_WORDS]
        text = ' '.join(filtered_words)
    
    return text

def extract_merchant_name(description: str) -> str:
    """
    Extract the primary merchant name from a transaction description.
    Uses heuristics to identify the most likely merchant.
    
    Args:
        description: Transaction description
        
    Returns:
        Extracted merchant name
    """
    # Clean the description first
    clean_desc = clean_transaction_text(description)
    
    # Split into words
    words = clean_desc.split()
    
    # If we have no words after cleaning, return unknown
    if not words:
        return "unknown"
    
    # Simple heuristic: first word is often the merchant
    merchant = words[0]
    
    # Check if we have a standardized merchant name
    for standard_name in set(MERCHANT_ALIASES.values()):
        if standard_name in clean_desc:
            return standard_name
    
    # If the merchant name is too short, try the first two words
    if len(merchant) <= 2 and len(words) > 1:
        merchant = ' '.join(words[:2])
    
    return merchant

def find_similar_transactions(query_desc: str, 
                             descriptions: List[str],
                             threshold: float = 0.7) -> List[int]:
    """
    Find similar transactions using simple string similarity.
    
    Args:
        query_desc: Query description
        descriptions: List of descriptions to search
        threshold: Similarity threshold (0-1)
        
    Returns:
        Indices of similar descriptions
    """
    from difflib import SequenceMatcher
    
    # Clean the query description
    clean_query = clean_transaction_text(query_desc)
    
    similar_indices = []
    
    for i, desc in enumerate(descriptions):
        # Clean the candidate description
        clean_desc = clean_transaction_text(desc)
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, clean_query, clean_desc).ratio()
        
        if similarity >= threshold:
            similar_indices.append(i)
    
    return similar_indices

def get_description_word_frequencies(descriptions: List[str]) -> Dict[str, int]:
    """
    Get word frequencies across all descriptions.
    Useful for identifying common terms and potential noise words.
    
    Args:
        descriptions: List of transaction descriptions
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    word_counts = {}
    
    for desc in descriptions:
        # Clean the description
        clean_desc = clean_transaction_text(desc)
        
        # Count words
        for word in clean_desc.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    return word_counts


# Example usage
if __name__ == "__main__":
    # Test transaction descriptions
    test_descriptions = [
        "COSTCO.CA TRANSACTION",
        "AMZN Mktp US*123ABC456",
        "MCDONALDS #1234",
        "NETFLIX.COM 123456789",
        "WALMART POS PURCHASE TERMINAL #1234",
        "ATM WITHDRAWAL BANK OF AMERICA #1234",
        "POS DEBIT PURCHASE AT SWISS CHALET 959",
    ]
    
    print("Original vs Cleaned Descriptions:")
    for desc in test_descriptions:
        clean = clean_transaction_text(desc)
        merchant = extract_merchant_name(desc)
        print(f"Original: {desc}")
        print(f"Cleaned:  {clean}")
        print(f"Merchant: {merchant}")
        print()
    
    # Get word frequencies
    word_freqs = get_description_word_frequencies(test_descriptions)
    print("\nMost common words:")
    for word, count in sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{word}: {count}")
    
    # Find similar transactions
    query = "COSTCO ONLINE PURCHASE"
    similar = find_similar_transactions(query, test_descriptions)
    print(f"\nTransactions similar to '{query}':")
    for idx in similar:
        print(f"- {test_descriptions[idx]}")
