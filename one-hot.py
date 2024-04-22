import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

chunk_size = 1000
one_hot_encoded_chunks = []


def smiles_to_one_hot(smiles, vocab_size):
    unique_chars = set(smiles)
    char_to_index = {char: i for i, char in enumerate(unique_chars)}
    one_hot_smiles = np.zeros((len(smiles), vocab_size), dtype=np.float32)
    for i, char in enumerate(smiles):
        one_hot_smiles[i, char_to_index[char]] = 1
    return one_hot_smiles


def process_chunk(chunk):
    one_hot_encoded_chunk = {}
    print('CPU Count => ', multiprocessing.cpu_count())
    for col in ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']:
        one_hot_encoded_col = Parallel(n_jobs=4)(
            delayed(smiles_to_one_hot)(smiles, len(set(smiles))) for smiles in chunk[col])
        one_hot_encoded_chunk[col] = one_hot_encoded_col
    return one_hot_encoded_chunk


data_parquet = pd.read_parquet("dataset/train.parquet")

# Dividir el parquet en chunks
num_chunks = len(data_parquet) // chunk_size + 1
chunks = np.array_split(data_parquet, num_chunks)

for chunk in chunks:
    one_hot_encoded_chunks.append(process_chunk(chunk))

# Obtener el tamaño máximo
max_lengths = {col: max(max(map(len, chunk[col])) for chunk in one_hot_encoded_chunks) for col in
               ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']}

# Ajustar el tamaño de los arrays one-hot según la longitud máxima encontrada
for chunk in one_hot_encoded_chunks:
    for col in chunk.keys():
        for smiles_array in chunk[col]:
            vocab_size = smiles_array.shape[1]  # Obtenemos el tamaño del vocabulario desde la primera secuencia
            n = max_lengths[col] - len(smiles_array)
            if n > 0:
                smiles_array = np.pad(smiles_array, ((0, n), (0, 0)), mode='constant', constant_values=0)

concatenated_chunks = {col: np.concatenate([chunk[col] for chunk in one_hot_encoded_chunks], axis=0) for col in
                       one_hot_encoded_chunks[0].keys()}

for col, array in concatenated_chunks.items():
    print(f"Dimensiones del array one-hot para la columna '{col}':", array.shape)
