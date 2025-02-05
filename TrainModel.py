# Import necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Create a WordNetLemmatizer instance
lemmatizer = WordNetLemmatizer()

# Load the intents file
intents = json.loads(open('intents.json').read())

# Initialize empty lists to store words, classes, and documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Loop through each intent and pattern to tokenize words and create documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and normalize words, and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Remove duplicates from classes
classes = sorted(set(classes))

# Serialize words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
outputEmpty = [0] * len(classes)

# Loop through each document to create bag-of-words representation and output row
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

# Split training data into input features and output labels
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Create a neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Define the optimizer and compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(np.array(trainX), np.array(trainY), verbose=1)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
model.save('chatbot_model.keras', hist)
print('Done')
