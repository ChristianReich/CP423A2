from preprocessing import preprocess_text, process_docs
from index import create_index
import os

from pathlib import Path

def search(query, index):
    """
    Search for a query in the index and return the results.
    
    Args:
        query (str): The search query.
        index (dict): The inverted index.
    
    Returns:
        dict: A dictionary with tokens as keys and their counts and document IDs as values.
    """
    results = []
    query_tokens = preprocess_text(query)
    
    # Finds all the valid document that contain all the tokens in the query
    valid_docs = None
    for token in query_tokens:
        if token in index:
            if valid_docs is None:
                valid_docs = set(index[token][1].keys())
            else:
                valid_docs.intersection(index[token][1].keys())
        else:
            print(f"Token '{token}' not found in index.")
            break
    
    # Contrsucts temporary index with only query tokens and valid documents
    temp_ind = {}
    if valid_docs is not None:
        for token in query_tokens:
            temp_ind[token] = [index[token][0], {}]
            for doc_id in index[token][1].keys():
                if doc_id in valid_docs:
                    temp_ind[token][1][doc_id] = index[token][1][doc_id]
    print(temp_ind)

    intial = temp_ind[query_tokens[0]][1]

    for doc_id, positions in intial.items():
        for pos in positions:
            flag = True
            for token in query_tokens[1:]:
                if doc_id in temp_ind[token][1] and ((pos + 1) in temp_ind[token][1][doc_id]):
                    pos += 1
                else:
                    flag = False
                    break
            if flag:
                results.append(doc_id)
    return results

def main():
    # Path to the current file
    current_file = Path(__file__).resolve()

    # Folder in the same directory
    target_folder = current_file.parent / 'data'

    # Process the documents in the target folder
    docs = process_docs(target_folder)

    # Create the index from the processed documents
    index = create_index(docs)
    flag = True
    while flag:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            flag = False
            continue
        
        else:
            results = search(query, index)
        
        # Display the results
        if results:
            print("Search Results:")
            print(results)
        else:
            print("No results found.")

    return

if __name__ == "__main__":
    main()