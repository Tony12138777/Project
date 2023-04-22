import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, BatchNormalization, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv', sep='\t')
train_data = train_data[['review', 'rating']]
train_data = train_data.dropna()

# Load the testing dataset
test_data = pd.read_csv('drugsCom_raw/drugsComTest_raw.tsv', sep='\t')
test_data = test_data[['review', 'rating']]
test_data = test_data.dropna()

# Encode the labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_data['rating'])
test_labels = encoder.transform(test_data['rating'])

# Tokenize the data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['review'])
train_sequences = tokenizer.texts_to_sequences(train_data['review'])
test_sequences = tokenizer.texts_to_sequences(test_data['review'])

# Pad the sequences
max_length = 1000
train_data = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_data = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Build the CNN model
model = Sequential()

# Add embedding layer
model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_length))

# Add 1D Convolutional layer with 64 filters and kernel size 5
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())

# Add Dropout layer to reduce overfitting
model.add(Dropout(0.3))

# Add Max Pooling layer with pool size 2
model.add(MaxPooling1D(pool_size=2))

# Add another 1D Convolutional layer with 128 filters and kernel size 5
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())

# Add Dropout layer to reduce overfitting
model.add(Dropout(0.3))

# Add Max Pooling layer with pool size 2
model.add(MaxPooling1D(pool_size=2))

# Add LSTM layer with 64 units and dropout of 0.3
model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.3))

# Add a fully connected layer with 512 units and ReLU activation
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())

# Add Dropout layer to reduce overfitting
model.add(Dropout(0.5))

# Add output layer with sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
