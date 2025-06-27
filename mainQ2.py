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

    tf_dfs = [binary_df, count_df, norm_df, log_df, dubnorm_df]
    total_docs = count_df.shape[0]
    df = (count_df > 0).sum(axis=0)
    idf = np.log(total_docs / (df + 1))
    tfidf_dfs = [tf_df.mul(idf, axis=1) for tf_df in tf_dfs]

    flag = True
    while flag:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            flag = False
            continue
        query_tokens = preprocess_text(query)

        query_vector = pd.DataFrame(0, columns=tfidf_dfs[0].columns, index=['Binary', 'Count', 'Norm', 'Log', 'DubNorm'], dtype=float)
        for token in query_tokens:
            if token in tfidf_dfs[0].columns:
                query_vector.at['Binary', token] = 1
                query_vector.at['Count', token] += 1
        query_vector.loc['Norm'] = query_vector.loc['Count'] / query_vector.loc['Count'].sum()
        query_vector.loc['Log'] = np.log1p(query_vector.loc['Count'])
        df = 0.5 + 0.5 * (query_vector.loc['Count'].div(query_vector.loc['Count'].max()))
        query_vector.loc['DubNorm'] = 0.5 + 0.5 * (query_vector.loc['Count'].div(query_vector.loc['Count'].max()))
        query_vector.mul(idf, axis=1)

        relevance = []
        for i in range(tfidf_dfs[0].shape[0]):
            scores = []
            for j in range(len(tfidf_dfs)):
                scores.append(tfidf_dfs[j].iloc[i].dot(query_vector.iloc[j]))
            relevance.append(scores)
        
        final_scores = pd.DataFrame(relevance, index=tfidf_dfs[0].index, columns=['Binary', 'Count', 'Norm', 'Log', 'DubNorm'])
        
        # Print all the top 5 results for each TF-IDF type
        print("Top 5 results for each TF-IDF type:")
        for col in final_scores.columns:
            print(f"\nTop 5 results for {col}:")
            print(final_scores.sort_values(by=col, ascending=False)[col].head(5))
        

    
    return

if __name__ == "__main__":
    main()