import random
import nltk
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

intents = json.loads(open("intent.json").read())
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words]
words = sorted(set(words))
classes = sorted(set(classes))

# Dump the words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Model training 
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns]

    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_X = list(training[:, 0])
train_y = list(training[:, 1])   

# Creating a model
model = Sequential()

model.add(Dense(256, input_shape = (len(train_X[0]), ), activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(len(train_y[0]), activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

callback_cb = tf.keras.callbacks.ModelCheckpoint("chatbot.h5", save_best_only = True)

# Fit the model
model_history = model.fit(np.array(train_X), np.array(train_y), epochs=200, batch_size=4, validation_data=(np.array(train_X), np.array(train_y)), callbacks=[callback_cb])

print("\n\nSUCCESSFULLY TRAINED!\n")