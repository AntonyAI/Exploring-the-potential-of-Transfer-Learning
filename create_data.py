import numpy as np
import pandas as pd
import pickle

#carico il file con gli embedding glove
#inizializzo per prima cosa il padding token
word2idx = {} 
word2idx['<pad>'] = 0
words = ['<pad>']
vectors = [np.zeros(200)]
#ora loop su tutte le parole
idx = 1
with open('glove.6B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)


# salvo il word index
with open('word_index.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

# creo un dict in cui associo ciascun token al suo word vector
glove = {}
for i, vect in enumerate(vectors):
    glove[words[i]] = vect


#ora salvo in pickle
with  open('glove_wordvectors.pkl', 'wb') as f:
    pickle.dump(glove, f)



