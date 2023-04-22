import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('DIAGNOSES_ICD.csv')

# Filter out records without a diagnosis
data = data[data.icd9_code.notnull()]

# Map ICD9 codes to text descriptions
icd9_codes = pd.read_csv('icd9_codes.tsv', delimiter='\t')
icd9_map = dict(zip(icd9_codes.DIAGNOSIS_CODE, icd9_codes.LONG_DESCRIPTION))
data['TEXT'] = data['icd9_code'].map(icd9_map)

# Convert diagnoses into binary labels
data['LABEL'] = 1

# Split into training and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Filter out records with NaN values in the 'TEXT' column
train = train[train['TEXT'].notna()]
print(train)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train['TEXT'].fillna("<OOV>"))

# Convert text to sequences of integers
X_train = tokenizer.texts_to_sequences(train['TEXT'].fillna("<OOV>"))
X_test = tokenizer.texts_to_sequences(test['TEXT'].fillna("<OOV>"))

# Pad sequences to a fixed length
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Create the model
inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=maxlen)(inputs)
conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)
dense_layer_1 = Dense(10, activation='relu')(pooling_layer)
dropout_layer = Dropout(0.2)(dense_layer_1)
dense_layer_2 = Dense(1, activation='sigmoid')(dropout_layer)
model = Model(inputs=inputs, outputs=dense_layer_2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, train['LABEL'], epochs=10, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, test['LABEL'], verbose=False)
print(f'Test Accuracy: {accuracy:.4f}')