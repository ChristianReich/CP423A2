from preprocessing import preprocess_text, process_docs
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def main():
    # Path to the current file
    current_file = Path(__file__).resolve()

    # Folder in the same directory
    target_folder = current_file.parent / 'data'

    # Process the documents in the target folder
    docs = process_docs(target_folder)
    # Keys and token lists
    doc_names = list(docs.keys())
    text_data = [' '.join(tokens) for tokens in docs.values()] # CountVectorizer expects a list of strings

    # Initialize vectorizer with binary mode
    vectorizer = CountVectorizer(binary=True)

    # Fit and transform
    X = vectorizer.fit_transform(text_data)

    # Convert to DataFrame
    df_binary = pd.DataFrame(X.toarray(), index=doc_names, columns=vectorizer.get_feature_names_out())
    return

if __name__ == "__main__":
    main()