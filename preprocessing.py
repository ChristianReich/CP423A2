import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(p):
    p = p.lower()  # Convert to lowercase
    tokens = word_tokenize(p) # Tokenize the text
    # Remove all stop words from token list
    tokens = [w for w in tokens if not w in STOP_WORDS] 
    # Remove punctuation and non-alphanumeric tokens
    cleaned_tokens = []
    for token in tokens:
        cleaned_token = re.sub(r'[^a-zA-Z0-9]', '', token)
        if len(cleaned_token) > 1:
            cleaned_tokens.append(cleaned_token)
    tokens = cleaned_tokens
    return tokens

def process_docs(path):
    processed_docs = {}
    vocab = []
    for filename in os.listdir(path):
        print(f"Processing file: {filename[:-4]}")
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                processed_docs[filename[:-4]] = preprocess_text(content)
    return processed_docs