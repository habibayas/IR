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
        tokenized_doc = [term for term in word_tokenize(content) if term.lower() not in stopwords.words('english')]
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
    final_token_list = [term for term in word_tokenize(content) if term.lower() not in stopwords.words('english')]

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
            prepared_doc.append(term)
    return prepared_doc
def query_optimizing(query):
    lis = [[] for _ in range(10)]
    for term in query.split():
        if term in pos_index:
            for key in pos_index[term][1].keys():
                if lis[key - 1] != [] and lis[key - 1][-1] == pos_index[term][1][key][0] - 1:
                    lis[key - 1].append(pos_index[term][1][key][0])
                else:
                    lis[key - 1].append(pos_index[term][1][key][0])

    positions = []
    for pos, lst in enumerate(lis, start=1):
        if len(lst) == len(query.split()):
            positions.append(f'document {pos}')
    return positions

def handle_boolean_operators(query):
    # Split the query into terms
    terms = query.split()

    # Initialize result as all documents
    result = set(range(1, 11))

    # Variables to keep track of current operator and term
    current_operator = 'AND'
    current_term = None

    # Loop through each term in the query
    for term in terms:
        if term.upper() in ['AND', 'OR', 'NOT']:
            # Update the current operator
            current_operator = term.upper()
        else:
            # Update the current term
            current_term = term.lower()  # Convert to lowercase

            # Apply the current operator to update the result set
            term_documents = set()
            if current_term in pos_index:
                term_documents = set(pos_index[current_term][1].keys())

            if current_operator == 'AND':
                result = result.intersection(term_documents)
            elif current_operator == 'OR':
                result = result.union(term_documents)
            elif current_operator == 'NOT':
                result = result.difference(term_documents)

    # Convert the result set to a list for better presentation
    result = list(result)
    result.sort()

    return result


def get_similarity(q, df, tf, wighted_tf, tdf, tf_idf, doc_len, norm_tf_idf):
    print("query:", q)
    query_terms = q.split()

    # Filter out boolean operators
    query_terms = [term for term in query_terms if term not in ['AND', 'OR', 'NOT']]

    q_vec = vectorizer.transform([' '.join(query_terms)]).toarray().reshape(df.shape[0], )
    sim = {}

    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    for doc, score in sim_sorted:
        if score > 0.5:
            print("similarity value:", score)
            print("the document is ", doc + 1)

    # Print other information as before
    print('\n f-raw')
    try:
        print(tf[query_terms])
    except KeyError as e:
        print(f"Error printing f-raw: {e}")

  # Print wighted_tf(1+ log tf)
    print('\n wighted_tf(1+ log tf)')
    for term in query_terms:
        if term in wighted_tf.index:
            print(wighted_tf.loc[term])
        else:
            print(f"{term}: Term not found in the DataFrame.")

# Print idf (modified to handle missing terms)
    print('\n idf')
    try:
        print(tdf.loc[query_terms])
    except KeyError as e:
        missing_terms = set(query_terms) - set(tdf.index)
    print(f"Error printing idf: Terms not found in DataFrame - {missing_terms}")


    print('\n tf*idf')
    print(tf_idf[query_terms])

    print('\n normalized')
    print(norm_tf_idf[query_terms])

all_terms = []
for doc in documents:
    for term in doc.split():
        all_terms.append(term)
all_terms = set(all_terms)
def get_tf(document):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        wordDict[word] += 1
    return wordDict

tf = pd.DataFrame(get_tf(documents[0]).values(), index=get_tf(documents[0]).keys())
for i in range(1, len(documents)):
    tf[i] = get_tf(documents[i]).values()
tf.columns = ['doc'+str(i) for i in range(1, 11)]
print(tf)
def get_wighted_tf(x):
    if x > 0:
        return math.log(x) + 1
    return 0

wighted_tf = tf.applymap(get_wighted_tf)
print(wighted_tf)
tdf = pd.DataFrame(columns=['df', 'idf'])
print("\n tdf \n")
print(tdf)
for i in range(len(tf)):
    inverse_term = wighted_tf.iloc[i].values.sum()
    tdf.loc[i, 'df'] = inverse_term
    tdf.loc[i, 'idf'] = math.log10(10 / (float(inverse_term)))
tdf.index = wighted_tf.index

tf_idf = wighted_tf.multiply(tdf['idf'], axis=0)
print("\n TF-IDF \n")
print(tf_idf)
def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

doc_len = pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc[0, col+'_length'] = get_doc_len(col)
print("\n doc_len \n")
print(doc_len)
def get_norm_tf_idf(col, x):
    try:
        return x / doc_len[col+'_length'].values[0]
    except:
        return 0

norm_tf_idf = tf_idf.apply(lambda x: get_norm_tf_idf(x.name, x), axis=0)
print(norm_tf_idf)
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(documents)
x = x.T.toarray()
df = pd.DataFrame(x, index=vectorizer.get_feature_names_out())

user_query = input("Enter your query: ")

# Check if boolean operators are present
boolean_operators = ['AND', 'OR', 'NOT']
if any(operator in user_query.upper() for operator in boolean_operators):
    # Handle boolean operators
    boolean_result = handle_boolean_operators(user_query)
    print("Boolean Query Result:")
    if boolean_result:
        for doc_id in boolean_result:
            print(f'Document {doc_id}')
    else:
        print("No matching documents found.")
else:
    # Print the positional index for the entire query
    print("\n Positonal Index for Input Query \n")
    for term in user_query.split():
        term_lower = term.lower()
        if term_lower in pos_index:
            print(f"{term_lower}: {pos_index[term_lower]}")
            # Call get_similarity for each term in the query
            get_similarity(term_lower, df, tf, wighted_tf, tdf, tf_idf, doc_len, norm_tf_idf)
        else:
            print(f"{term_lower}: Term not found in the positional index.")
