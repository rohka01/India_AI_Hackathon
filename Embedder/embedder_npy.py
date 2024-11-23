import pandas as pd
import numpy as np
import tqdm.notebook as tq
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import pickle

dataset = pd.read_csv('./data/super_df.csv', low_memory=False)
labelencoder = LabelEncoder()
#dataset[:, 'super_category'] = labelencoder.fit_transform(dataset[:, 'super_category'])
dataset['super_category'] = labelencoder.fit_transform(dataset['super_category'])
dataset = dataset[~dataset['crimeaditionalinfo'].isna()]
dataset = dataset[dataset['crimeaditionalinfo'] != ' ']
#Dataset 
X = list(dataset['crimeaditionalinfo'])
y = list(dataset['super_category'])

#embeddings = OllamaEmbeddings(model="mistral-nemo") # You can try 
embeddings = OllamaEmbeddings(model="llama2-uncensored") # You can try 

def embed_text(text):
    single_vector = embeddings.embed_query(text)
    return single_vector

# Output file path for npy file
npy_file = './data/super_df_embd.npy'

# Prepare lists to hold your data
X_embd_list = []
X_embd_val_list = []
y_embd_list = []
y_class_list = []

# Function to save the collected data incrementally
def save_incrementally():
    final_data = {
        'Embedding': np.array(X_embd_list, dtype=object),
        'Entries': np.array(X_embd_val_list),
        'Encoded Class': np.array(y_embd_list),
        'Original Class': np.array(y_class_list)
    }
    
    # If the file exists, append the data, else create a new file
    if os.path.exists(npy_file):
        # Load the existing data from the file
        existing_data = np.load(npy_file, allow_pickle=True).item()
        
        # Append the new data to existing data
        for key in final_data.keys():
            existing_data[key] = np.concatenate((existing_data[key], final_data[key]), axis=0)
        
        # Save back to the file
        np.save(npy_file, existing_data)
    else:
        # If the file doesn't exist, create it
        np.save(npy_file, final_data)

# Iterate and collect the data
for i in tqdm(range(len(X))):
    # Get embedding and labels for the current entry
    X_embd = embeddings.embed_query(X[i])
    X_embd_val = X[i]
    y_embd = y[i]
    y_class = labelencoder.classes_[y[i]]

    # Append the new data to the lists
    X_embd_list.append(X_embd)
    X_embd_val_list.append(X_embd_val)
    y_embd_list.append(y_embd)
    y_class_list.append(y_class)

    # Save the collected data incrementally after each iteration
    save_incrementally()

    # Clear lists to free up memory after saving
    X_embd_list.clear()
    X_embd_val_list.clear()
    y_embd_list.clear()
    y_class_list.clear()