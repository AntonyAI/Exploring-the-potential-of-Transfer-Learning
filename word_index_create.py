# # Creazione word index
from transformers import BertTokenizer
import pandas as pd
import pickle

train_raw = pd.read_csv('Train.csv')

#e il validation set
valid_raw = pd.read_csv('Valid.csv')


#instanzio il tokenizer di BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#genero gli indici dei tokens del testo di input
inputs_train = tokenizer(train_raw['text'].tolist(), max_length= 256, padding= 'max_length', truncation= True, return_tensors= 'pt')

# appendo al dizionario anche la label
inputs_train['labels'] = torch.tensor(train_raw['label'].tolist(), dtype= torch.float)

#..stessa cosa per il validation set
inputs_val = tokenizer(valid_raw['text'].tolist(), max_length= 256, padding= 'max_length', truncation= True, return_tensors= 'pt')

inputs_val['labels'] = torch.tensor(valid_raw['label'].tolist(), dtype= torch.float)

#ora salvo i due inputs in pickle
with open('inputs_train.pkl', 'wb') as f:
    pickle.dump(inputs_train, f)

# %%
with open('inputs_val.pkl', 'wb') as f:
    pickle.dump(inputs_val, f)
