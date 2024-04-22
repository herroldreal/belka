import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Chunk file
chunk_size = 10 ** 6
chunk_list = []
chunk_test_list = []
chunk_result = []

# Mapping data types
dtype_mapping_test = {
    'id': np.int64,
    'buildingblock1_smiles': str,
    'buildingblock2_smiles': str,
    'buildingblock3_smiles': str,
    'molecule_smiles': str,
    'protein_name': str
}

dtype_mapping_train = {
    'id': np.int64,
    'buildingblock1_smiles': str,
    'buildingblock2_smiles': str,
    'buildingblock3_smiles': str,
    'molecule_smiles': str,
    'protein_name': str,
    'binds': np.int64
}


def process_chunk(chunk):
    # train_data = pd.concat(chunk_result, ignore_index=True)
    # print(train_data.head())
    #print(train_data['binds'].value_counts())
    # print('Train data => ', train_data)
    # x = train_data.drop(columns=['id', 'protein_name', 'binds'])
    # y = train_data['binds']

    print('Chunk => ', chunk)

    x = chunk.drop(columns = ['id', 'protein_name', 'binds'])
    y = chunk['binds']

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy", accuracy)


# Load test data and training data
for chunk in pd.read_csv('dataset/train.csv', chunksize=chunk_size):
    process_chunk(chunk)

# for chunkTest in pd.read_csv('dataset/test.csv', chunksize=10000, dtype=dtype_mapping_test):
#    chunk_test_list.append(chunkTest)

# train_data = pd.concat(chunk_result, ignore_index=True)
# print(train_data.head())
# print(train_data['binds'].value_counts())
