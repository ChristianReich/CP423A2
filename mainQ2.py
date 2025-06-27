from preprocessing import preprocess_text, process_docs
from pathlib import Path
import pandas as pd
import numpy as np
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
    vectorizer_binary = CountVectorizer(binary=True)

    # Fit and transform
    X_binary = vectorizer_binary.fit_transform(text_data)

    # Convert to DataFrame
    binary_df = pd.DataFrame(X_binary.toarray(), index=doc_names, columns=vectorizer_binary.get_feature_names_out())

    vectorizer_count = CountVectorizer()
        # Fit and transform
    X_count = vectorizer_count.fit_transform(text_data)

    # Convert to DataFrame
    count_df = pd.DataFrame(X_count.toarray(), index=doc_names, columns=vectorizer_count.get_feature_names_out())
    
    norm_df = count_df.div(count_df.sum(axis=1), axis=0)
    log_df = np.log1p(count_df)
    dubnorm_df = 0.5 + 0.5 * (count_df.div(count_df.max(axis=1), axis=0))
    print(dubnorm_df[dubnorm_df["00"] > 0.5].head())
    return

if __name__ == "__main__":
    main()