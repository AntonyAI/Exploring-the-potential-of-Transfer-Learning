# Exploring-the-potential-of-Transfer-Learning
## Project Details
The aim of the project is to show how can be exploited the potential of Transfer Learning in the field of NLP.\
The particular Use Case is a Sentiment Analysis project, with the famous IMDB Dataset. You can easily load it from the Kaggle website at [this](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset).\
The project is organized as follows:\
1.\
    - in the file *create_data.py* it has been loaded and saved the word index using the **GLOVE Embeddings**;\
    - the subsequent step is to use these pretrained word embedding in order to set up a a Bidirectional LSTM in the *lstm_training.py* file.\
2. \
    - in the file *word_index_create.py*, instead, it has been exploited the tokenizer of *BERT* in order to create the word index for the subsequent evaluation;\
    - in the file *bert_training.py* it has been showed how it can be exploited the representation of the last layer of BERT, in order to to build up on top of that a series of custom layers.

In this case, the gradient of BERT parameters has been freezed in order to not update those in the Backpropagation.
Lastly, the notebook *visualization.ipynb* shows up the core metrics during the learning process for both models.

We can see that with GLOVE+LSTM we were able to achieve 85% accuracy on validation set, while with BERT+LSTM combo, we have managed to achieve a 91% accuracy on validation set.
