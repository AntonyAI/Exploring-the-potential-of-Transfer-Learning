import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 12, 8
plt.style.use("ggplot")
from tqdm import tqdm  # for progress bar


# userò l'imdb dataset scaricato da kaggle
train_raw = pd.read_csv("Train.csv")

# e il validation set
valid_raw = pd.read_csv("Valid.csv")


# # Caricamento dati e creazione dataset

with open("inputs_train.pkl", "rb") as f:
    inputs_train = pickle.load(f)

with open("inputs_val.pkl", "rb") as f:
    inputs_val = pickle.load(f)

# Ora passiamo alla creazione dei dataset e dei dataloader


class CustomDataset(Dataset):
    def __init__(self, input_encodings):
        self.input_encodings = input_encodings

    def __getitem__(self, idx):
        # mi ritorna il dizionario contenente tutti gli elementi corrispondenti a quell'idx
        return {key: val[idx] for key, val in self.input_encodings.items()}

    def __len__(self):
        return len(self.input_encodings["input_ids"])


train_dataset = CustomDataset(input_encodings=inputs_train)
val_dataset = CustomDataset(input_encodings=inputs_val)

# ora creo i dataloader: batch size la fisso inizialmente a 32
BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)


# # Creazione del modello

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# aggiungeremo al layer di bert uno strato lstm e due strati fully connected
class BertClassifier(nn.Module):
    def __init__(
        self,
        hidden_size_lstm,
        hidden_size_fc,
        dropout_rate_lstm=0.25,
        dropout_rate_fc=0.3,
    ):
        super(BertClassifier, self).__init__()

        # andiamo a creare i vari strati
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size_lstm,
            num_layers=1,
            dropout=dropout_rate_lstm,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate_fc)
        self.fc = nn.Linear(hidden_size_lstm * 2, hidden_size_fc)
        self.out = nn.Linear(hidden_size_fc, 1)

    # forward pass
    def forward(self, input_ids, attention_mask):
        # output di bert
        bert_out, _ = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        # lstm layer
        lstm = self.lstm(bert_out)
        # mean pooling
        pooler_lstm_out = lstm[0].mean(dim=1)
        # max pooling
        pooler_lstm_max = lstm[0].max(dim=1)[0]
        # concateno
        final_pooler = torch.cat((pooler_lstm_out, pooler_lstm_max), dim=1)
        # fully connected
        lin = self.fc(final_pooler)
        # elu activation
        lin = torch.nn.functional.elu(lin)
        # dropout
        drop_out = self.dropout(lin)
        # output
        out = self.out(drop_out)

        return out


# instanzia il modello
model = BertClassifier(hidden_size_lstm=32, hidden_size_fc=16).to(device=device)

# non vogliamo addestrare i parametri del layer di bert
for param in model.bert.parameters():
    param.requires_grad = False

# funzione di perdita
loss_function = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler cosinusoidale per il learning rate, il quale avrà dunque andamento cicliclo
scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, eta_min=3e-5, T_0=100)

# training per 5 epoche
EPOCHS = 5
train_loss = []
train_acc = []
val_loss = []
val_acc = []
obs_train, pred_train = [], []
obs_val, pred_val = [], []
for epoch in range(EPOCHS):
    # training step
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_batch = batch["labels"].to(device)
        # forward prop
        optimizer.zero_grad()

        model_out = model(input_ids, attention_mask)

        y_pred = sigmoid(model_out.squeeze())

        # criterio
        loss = loss_function(y_pred, y_batch)
        # appendo il valore della loss
        train_loss.append(loss.item())
        # previsioni e osservazioni
        preds = y_pred.cpu().detach().numpy().round()
        obs = y_batch.cpu().detach().numpy()
        # prendo previsioni e osservazioni e le storo
        pred_train.append(preds)
        obs_train.append(obs)
        # accuracy
        train_acc.append(accuracy_score(y_true=obs, y_pred=preds))

        # backward prop
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    # validation step
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_batch = batch["labels"].to(device)
            # forward prop
            model_out = model(input_ids, attention_mask)

            y_pred = sigmoid(model_out.squeeze())
            # criterio
            loss = loss_function(y_pred, y_batch)
            # appendo la loss
            val_loss.append(loss.item())
            # previsioni e osservazioni
            preds = y_pred.cpu().detach().numpy().round()
            obs = y_batch.cpu().detach().numpy()
            # prendo previsioni e osservazioni e le storo
            pred_val.append(preds)
            obs_val.append(obs)
            # accuracy
            val_acc.append(accuracy_score(y_true=obs, y_pred=preds))

# visualizzare il processo di apprentimento
fig, ax = plt.subplots(2, 1)
ax[0].plot(train_loss, color="red")
ax[1].plot(train_acc, color="darkgreen")
plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(val_loss, color="red")
ax[1].plot(val_acc, color="darkgreen")
plt.show()


# torch.save(model.state_dict(), "./models/bert_classifier.pt")  # save the model

# salvo le accuracy su train e validation set
with open("results/batch_train_acc_bert.pkl", "wb") as f:
    pickle.dump(train_acc, f)

with open("results/batch_dev_acc_bert.pkl", "wb") as f:
    pickle.dump(val_acc, f)

# show some log of the learning process
# steps_per_epoch_train = train_dataset.__len__() // BATCH_SIZE
# steps_per_epoch_val = val_dataset.__len__() // BATCH_SIZE

# for i in range(EPOCHS):
#     print(
#         "Epoch {} --> train accuracy: {}".format(
#             i, train_acc[(steps_per_epoch_train * i) : (i + 1) * steps_per_epoch_train]
#         )
#     )
#     print(
#         "Epoch {} --> validation accuracy: {}".format(
#             i, val_acc[(steps_per_epoch_val * i) : (i + 1) * steps_per_epoch_val]
#         )
#     )
