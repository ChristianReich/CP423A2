

def create_index(docs):
    index = {}
    for doc_id, tokens in docs.items():
        i = 0
        for token in tokens:  # tokens[1] contains the list of tokens
            if token not in index:
                index[token] = [0,{doc_id: []}]
            if doc_id not in index[token][1]:
                index[token][1][doc_id] = [i]
            else:
                index[token][1][doc_id].append(i) # add the position of the token in the document
            index[token][0] += 1  # increment the count of the token
            i += 1
    return index