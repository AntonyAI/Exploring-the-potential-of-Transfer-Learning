import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.figsize"] = 12, 8
plt.style.use("seaborn")

# Carico i word vectors e il word index, salvati in precedenza

with open("glove_wordvectors.pkl", "rb") as f:
    glove = pickle.load(f)

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

# Creo la matrice degli embeddings e successivamente inizializzo il layer di pytorch

embedding_matrix = np.zeros((len(word_index), 200))  # dim vocab x dim embedding
for i, item in enumerate(glove.items()):
    embedding_matrix[i, :] = item[1]

# padding dell'idx: servirà specificarlo a torch
idx_pad = word_index["<pad>"]

# inizializzo il tensore dell'embedding
embedding_tensor = torch.FloatTensor(embedding_matrix)

# creo il layer di embedding
embedding = nn.Embedding.from_pretrained(embedding_tensor, padding_idx=idx_pad)


# # Dataset loading e cleaning
# Ora carico il dataset IMDB e faccio il lavoro di pulizia
# userò l'imdb dataset scaricato da kaggle
train_raw = pd.read_csv("Train.csv")
train_raw.head()
# validation set
valid_raw = pd.read_csv("Valid.csv")
valid_raw.head()

# class per fare pre processing del testo: fondamentale per il modello
class TextPreProcess:
    def __init__(self, dati: pd.DataFrame, max_len: int):
        self.dati = dati
        self.max_len = max_len

    def cleaning(self, text):
        # tokenization
        tokens_list = word_tokenize(text)
        # stripping non alpha characters
        cleaned = []
        for token in tokens_list:
            if token.isalpha():
                cleaned.append(token.lower())

        return cleaned

    ###padding e troncamento delle sequenze###
    def truncate_and_pad_sequence(self) -> list:
        cleaned = self.dati.text.apply(lambda x: self.cleaning(text=x))
        # ora iterare su tutte le righe della lista tokens
        process_tokens = []
        for tokens in cleaned:
            if len(tokens) < self.max_len:
                true_tokens = tokens + ["<pad>"] * (self.max_len - len(tokens))
            else:
                true_tokens = tokens[: self.max_len]

            process_tokens.append(true_tokens)

        return process_tokens

    ###metodo che sostituisce i tokens non rientranti nel vocab come unk token###
    def get_idx(self, word_index: dict) -> np.array:
        # insieme di tutte le parole nel dict
        words = []
        for word in word_index.keys():
            words.append(word)

        # bisogna ora tirare fuori gli indici dai tokens
        tokens = self.truncate_and_pad_sequence()
        # inizializzo l'array
        idx = np.zeros((self.dati.shape[0], self.max_len))

        for i, token_list in tqdm(enumerate(tokens)):
            for j, token in enumerate(token_list):
                if token in words:
                    idx[i, j] = word_index[token]
                else:
                    idx[i, j] = word_index["<unk>"]

        return idx


preprocess_train = TextPreProcess(dati=train_raw, max_len=150)
preprocess_valid = TextPreProcess(dati=valid_raw, max_len=150)

tokens_id_train = preprocess_train.get_idx(word_index=word_index)
tokens_id_valid = preprocess_valid.get_idx(word_index=word_index)


# Ora prendere la y e convertire in tensore

y_train = torch.tensor(train_raw["label"])
y_valid = torch.tensor(valid_raw["label"])


# token indeces for train and validation sets
X_train = torch.tensor(tokens_id_train, dtype=torch.int)
X_valid = torch.tensor(tokens_id_valid, dtype=torch.int)

# Ora creeremo il dataset congeniale per il batching di pytorch
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# creation of the two datasets
train_dataset = CustomDataset(X=X_train, y=y_train)
valid_dataset = CustomDataset(X=X_valid, y=y_valid)

# CREO IL DATALOADER PER BATCHIFICARE IL DATASET
BATCH_SIZE = 64
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)


# # MOdel building

# simple bidirectional lstm
class BILSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0, bidirectional=True):
        super(BILSTM, self).__init__()
        # embedding layer da non trainare
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_tensor, padding_idx=idx_pad
        )
        # ora l'ltsm
        self.lstm = nn.LSTM(
            input_size=200,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # ora l'ultimo fully connected: ricordare il pooling prima di passare al fully connected
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        # passo alla lstm
        out, _ = self.lstm(embedded)
        # reshape dell'output per il layer finale: applico un mean pooling
        out_pool = out.mean(dim=1)
        # fully connected
        final_out = self.linear(out_pool)

        return final_out


# setting device to cuda (if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# instantiate the model
model = BILSTM(hidden_size=40, dropout=0.1).to(device)

# and the criterion, optimizer and the sigmoid activation for last layer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
sigmoid = nn.Sigmoid()

# start training: 20 epochs
EPOCHS = 20
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in tqdm(range(EPOCHS)):
    # training step
    model.train()
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        # forward prop
        model_out = model(X_batch)
        # sigmoide
        y_pred = sigmoid(model_out.squeeze())
        # criterio
        loss = criterion(y_pred, y_batch.float())
        train_loss.append(loss.item())
        # previsioni e osservazioni
        preds = y_pred.cpu().detach().numpy().round()
        obs = y_batch.cpu().detach().numpy()
        # accuracy
        train_acc.append(accuracy_score(y_true=obs, y_pred=preds))

        # forward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation step
    model.eval()
    for X_batch, y_batch in valid_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        # forward prop
        model_out = model(X_batch)
        # sigmoide
        y_pred = sigmoid(model_out.squeeze())
        # criterio
        loss = criterion(y_pred, y_batch.float())
        val_loss.append(loss.item())
        # previsioni e osservazioni
        preds = y_pred.cpu().detach().numpy().round()
        obs = y_batch.cpu().detach().numpy()
        # accuracy
        val_acc.append(accuracy_score(y_true=obs, y_pred=preds))

fig, ax = plt.subplots(2, 1)
ax[0].plot(train_loss)
ax[1].plot(train_acc)
plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(val_loss, color="orange")
ax[1].plot(val_acc, color="orange")
plt.show()

with open("./results/batch_train_acc_lstm.pkl", "wb") as f:
    pickle.dump(train_acc, f)

with open("./results/batch_dev_acc_lstm.pkl", "wb") as f:
    pickle.dump(val_acc, f)
