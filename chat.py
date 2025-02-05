import random
import json
import pickle
import numpy as np
import nltk
import string
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

# Load data files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the model
model = load_model('chatbot_model.keras')

# Load the TinyLlama model
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load the model
modelMama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Create the pipeline with both model and tokenizer
pipe = pipeline("text-generation", model=modelMama, tokenizer=tokenizer)
#pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Dictionary to hold context for each user
context = {}

def set_context(user_id, tag):
    context[user_id] = tag

def get_context(user_id):
    return context.get(user_id, "")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, user_id):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    # Add context handling
    if return_list and get_context(user_id) == return_list[0]['intent']:
        return return_list[1:] if len(return_list) > 1 else return_list
    return return_list

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def get_response(intents_list, intents_json, user_input):
    tag = intents_list[0]['intent']
    set_context('default', tag)  # Assuming single user for this example
    list_of_intents = intents_json['intents']
    
    user_input_normalized = normalize_text(user_input)
    
    for i in list_of_intents:
        if i['tag'] == tag:
            if i.get('deterministic', False):
                patterns_normalized = [normalize_text(pattern) for pattern in i['patterns']]
                if user_input_normalized in patterns_normalized:
                    pattern_index = patterns_normalized.index(user_input_normalized)
                    response = i['responses'][pattern_index]
                else:
                    response = "I'm sorry, I don't understand. Can you please rephrase?"
            else:
                response = random.choice(i['responses'])
            break
    return response

def generate_response_with_tinyllama(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable healthcare assistant who provides accurate and helpful medical information.",
        },
        {"role": "user", "content": user_input},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]
    return response

# Function to handle user input and responses
def chatbot_response(message, user_id):
    ints = predict_class(message, user_id)
    if ints:
        res = get_response(ints, intents, message)
        print(res)


print("Hello, Im here to answer your medical related questions")

while True:
    user_id = 'default'  # In a real application, use a unique identifier for each user
    message = input("").strip()
    if message.lower().startswith("tell me about"):
        condition = message.split("tell me about ")[-1]
        print(generate_response_with_tinyllama(condition))
    else:
        chatbot_response(message, user_id)