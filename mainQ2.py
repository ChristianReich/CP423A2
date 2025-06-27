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

        relevance = []
        for i in range(tfidf_dfs[0].shape[0]):
            scores = []
            for j in range(len(tfidf_dfs)):
                for token in query_tokens:
                    if token in tfidf_dfs[j].columns:
                        k = tfidf_dfs[j].columns.get_loc(token)
                        if token == query_tokens[0]:
                            scores.append(tfidf_dfs[j].iloc[i,k])
                        else:
                            scores[j] += tfidf_dfs[j].iloc[i,k]
            relevance.append(scores)
        
        final_scores = pd.DataFrame(relevance, index=tfidf_dfs[0].index, columns=['Binary', 'Count', 'Norm', 'Log', 'DubNorm'])
        print(final_scores.sort_values(by='Binary', ascending=False).head())
                
        


    # Uncomment the following lines if you want to create a final DataFrame with all TF-IDF values
    # tfidf_final_df = pd.DataFrame(index=count_df.index, columns=count_df.columns)

    # for doc in count_df.index:
    #     for term in count_df.columns:
    #         tfidf_final_df.at[doc, term] = [
    #             tfidf_df.at[doc, term] for tfidf_df in tfidf_dfs
    #         ]

    
    return

if __name__ == "__main__":
    main()