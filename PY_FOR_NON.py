import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer

dir_containing_files = r'E:\L4S1\IR\projct\files'
read_files = dir_containing_files

# Tokenize and remove stopwords
documents = []
for file_name in natsorted(os.listdir(read_files)):
    with open(os.path.join(read_files, file_name), 'r', encoding='utf-8') as f:
        content = f.read()
        tokenized_doc = [term.lower() for term in word_tokenize(content) if term.lower() not in stopwords.words('english')]
        documents.append(" ".join(tokenized_doc))
print(len(documents))

# Initialize the file no.
fileno = 1

# Initialize the dictionary.
pos_index = {}

# For every file.
for file_name in natsorted(os.listdir(read_files)):
    with open(os.path.join(read_files, file_name), 'r', encoding='utf-8') as f:
        content = f.read()

    # Tokenize and remove stopwords
    final_token_list = [term.lower() for term in word_tokenize(content) if term.lower() not in stopwords.words('english')]

    for pos, term in enumerate(final_token_list):
        if term in pos_index:
            # Increment total freq by 1.
            pos_index[term][0] += 1
            # Check if the term has existed in that DocID before.
            if fileno in pos_index[term][1]:
                pos_index[term][1][fileno].append(pos)
            else:
                pos_index[term][1][fileno] = [pos]
        else:
            # Initialize the list.
            pos_index[term] = [1, {fileno: [pos]}]

    # Increment the file no. counter for document ID mapping
    fileno += 1

# Print the positional index
for term, data in pos_index.items():
    print(f"{term}: {data}")

# Define the stemming function (assuming you have it defined somewhere in your code)
def stemming(doc):
    token_docs = word_tokenize(doc)
    stop_words = stopwords.words('english')
    stop_words.remove('in')
    stop_words.remove('to')
    stop_words.remove('where')

    prepared_doc = []
    for term in token_docs:
        if term not in stop_words:
            prepared_doc.append(term.lower())
    return prepared_doc

def query_optimizing(query):
    lis = [[] for i in range(10)]
    for term in query.split():
        term_lower = term.lower()
        if term_lower in pos_index.keys():
            for key in pos_index[term_lower][1].keys():
                if lis[key-1] != []:
                    if lis[key-1][-1] == pos_index[term_lower][1][key][0]-1:
                        lis[key-1].append(pos_index[term_lower][1][key][0])
                else:
                    lis[key-1].append(pos_index[term_lower][1][key][0])

    positions = []
    for pos, lst in enumerate(lis, start=1):
        if len(lst) == len(query.split()):
            positions.append('document '+str(pos))
    return positions

all_terms = []
for doc in documents:
    for term in doc.split():
        all_terms.append(term.lower())
all_terms = set(all_terms)

def get_tf(document):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        wordDict[word.lower()] += 1
    return wordDict

tf = pd.DataFrame(get_tf(documents[0]).values(), index=get_tf(documents[0]).keys())
for i in range(1, len(documents)):
    tf[i] = get_tf(documents[i]).values()
tf.columns = ['doc'+str(i) for i in range(1, 11)]

print("\n Term Frequency (TF) \n")
print(tf)

def get_wighted_tf(x):
    if x > 0:
        return math.log(x) + 1
    return 0

wighted_tf = tf.copy()
for i in range(0, len(documents)):
    wighted_tf['doc'+str(i+1)] = tf['doc'+str(i+1)].apply(get_wighted_tf)

print("\n Weighted Term Frequency (1 + log TF) \n")
print(wighted_tf)

tdf = pd.DataFrame(columns=['df', 'idf'])
for i in range(len(tf)):
    inverse_term = wighted_tf.iloc[i].values.sum()
    tdf.loc[i, 'df'] = inverse_term
    tdf.loc[i, 'idf'] = math.log10(10 / (float(inverse_term)))

tdf.index = wighted_tf.index

tf_idf = wighted_tf.multiply(tdf['idf'], axis=0)

def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

doc_len = pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc[0, col+'_length'] = get_doc_len(col)

print("\n Document Lengths \n")
print(doc_len)

def get_norm_tf_idf(col, x):
    try:
        return x / doc_len[col+'_length'].values[0]
    except:
        return 0

norm_tf_idf = pd.DataFrame()
for col in tf_idf.columns:
    norm_tf_idf[col] = tf_idf[col].apply(lambda x: get_norm_tf_idf(col, x))

print("\n Normalized TF-IDF \n")
print(norm_tf_idf)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(documents)
x = x.T.toarray()
df = pd.DataFrame(x, index=vectorizer.get_feature_names_out())

def get_similarity(q, df):
    print("query:", q)
    query = [q]
    q_vec = vectorizer.transform(query).toarray().reshape(df.shape[0],)
    sim = {}
    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    for doc, score in sim_sorted:
        if score > 0.5:
            print("similarity value:", score)
            print("the document is ", doc + 1)

    print('\n f-raw')
    print(tf.loc[q.split()])

    print('\n wighted_tf(1+ log tf)')
    print(wighted_tf.loc[q.split()])

    print('\n idf')
    print(tdf.loc[q.split()])

    print('\n tf*idf')
    print(tf_idf.loc[q.split()])

    print('\n normalized')
    print(norm_tf_idf.loc[q.split()])
# Example usage of the modified get_similarity function
user_query = input("Enter your query: ")
query_result = query_optimizing(user_query)
print("Query Result:")
if query_result:
    for position in query_result:
        print(position)
else:
    print("No matching documents found.")
# Print the positional index for the input query
print("\n Positonal Index for Input Query \n")
for term in user_query.split():
    term_lower = term.lower()
    if term_lower in pos_index:
        print(f"{term_lower}: {pos_index[term_lower]}")
    else:
        print(f"{term_lower}: Term not found in the positional index.")

# Call get_similarity outside the loop, passing the df parameter
get_similarity(user_query, df)
