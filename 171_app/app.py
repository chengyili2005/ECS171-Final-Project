# For backend functionality
from flask import Flask, send_from_directory, request, jsonify
import os
import sqlite3

# For language processing
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# For importing helper functions
print("Importing a special helper module, ignore warnings about unitialized classifier head weights")
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PARENT_DIR)
import make_features_helper as helper
print("Done importing that special module")

# For BERT
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import re
import joblib

# For XGBoost
import numpy as np
from nltk.tokenize import sent_tokenize
import pandas as pd

# For Neural Network
import tensorflow as tf

# For LSTM
import json
import torch

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

# Model paths
MODELS_DIR = 'models'

XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_classifier.joblib')
XGB_ENCODER_PATH = os.path.join(MODELS_DIR, 'xgb_label_encoder.joblib')

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_classifier.pt')
LSTM_STATUS2INT_PATH = os.path.join(MODELS_DIR, 'lstm_status2int_mapping.json')
LSTM_WORD2INT_PATH = os.path.join(MODELS_DIR, 'lstm_word2int_mapping.json')

BERT_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_classifier.pt')
BERT_ENCODER_PATH = os.path.join(MODELS_DIR, 'bert_label_encoder.joblib')

BINARY_NN_PATH = os.path.join(MODELS_DIR, 'neural_network_classifier.keras')

# Load models globally (placeholders - will fail until files exist)
models = {
    'xgboost': {},
    'lstm': {},
    'bert': {},
    'binary_nn': {}
}

def load_models():
    try:

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FIRST_PERSON = ['i', 'me', 'my', 'mine', 'myself']
        NEGATIVES = ['no', 'not', 'never', 'nothing', 'wrong', 'nope']
        SUICIDE_WORDS = ['die', 'end', 'forever', 'leave', 'gone', 'suicide', 'kill']
        RATIO_ITEMS = {'first_person_ratio' : FIRST_PERSON, 'negatives_ratio': NEGATIVES, 'suicide_ratio': SUICIDE_WORDS}
        FEATURE2FUNCTION = {
            'word_ratio' : helper.word_ratio,
            'reps_ratio' : helper.reps_ratio,
            'character_count' : helper.character_count,
            'word_count' : helper.word_count,
            'avg_word_length' : helper.avg_word_length,
            'avg_sentence_length_in_words' : helper.avg_sentence_length_in_words,
            'avg_sentence_length_in_characters' : helper.avg_sentence_length_in_characters,
            'sia_sentiment' : helper.sia_sentiment,
        }
        FEATURES = ['first_person_ratio',
        'negatives_ratio',
        'suicide_ratio',
        'reps_ratio',
        'character_count',
        'word_count',
        'avg_word_length',
        'avg_sentence_length_in_words',
        'avg_sentence_length_in_characters',
        'sia_sentiment',
        'statement']

        if os.path.exists(XGB_MODEL_PATH) and os.path.exists(XGB_ENCODER_PATH):

            # Initialize model
            models['xgboost']['preprocessor'], models['xgboost']['model'] = joblib.load(XGB_MODEL_PATH)
            models['xgboost']['label_encoder'] = joblib.load(XGB_ENCODER_PATH)

            # Define a forward pass function
            def forward_xgb(texts):
                vectors = []
                for text in texts:
                    vector = []
                    for column in FEATURE2FUNCTION.keys():
                        function = FEATURE2FUNCTION[column]
                        if column == 'word_ratio':
                            for ratio in RATIO_ITEMS.keys():
                                ratio_item = RATIO_ITEMS[ratio]
                                vector.append(function(text, ratio_item))
                        elif 'avg_sentence' in column:
                            sentences = sent_tokenize(text)
                            vector.append(function(sentences))
                        else:
                            vector.append(function(text))
                    vector.append(text)
                    temp = {}
                    for index, feature in enumerate(FEATURES):
                        temp[feature] = vector[index]
                    vector = pd.DataFrame([temp])
                    vector = models['xgboost']['preprocessor'].transform(vector)
                    vectors.append(vector)
                return vectors

            # Define an inference function
            def inference_xgb(texts):
                vectors = forward_xgb(texts)
                predictions = models['xgboost']['model'].predict(vectors[0])
                return models['xgboost']['label_encoder'].inverse_transform(predictions)[0]
            models['xgboost']['inference'] = inference_xgb
            print("XGBoost loaded successfully")

        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_STATUS2INT_PATH) and os.path.exists(LSTM_WORD2INT_PATH):

            # Initialize mappings
            with open(LSTM_WORD2INT_PATH, 'r') as f:
                models['lstm']['word2int'] = json.load(f)
            with open(LSTM_STATUS2INT_PATH, 'r') as f:
                models['lstm']['status2int'] = json.load(f)
            models['lstm']['int2status'] = {int(value): key for key, value in models['lstm']['status2int'].items()}

            # Initialize forward pass & model
            class LSTMClassifier(nn.Module):
                def __init__(self, vocab_size, embed_dim, hidden_dim, num_statuses):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
                    self.fc = nn.Linear(hidden_dim * 2, num_statuses)
                    self.dropout = nn.Dropout(0.3)

                def forward(self, x):
                    embeds = self.embedding(x)
                    embeds = self.dropout(embeds)
                    _, (hidden, _) = self.lstm(embeds)
                    hidden = torch.cat((hidden[0], hidden[1]), dim=1)
                    out  = self.fc(self.dropout(hidden))
                    return out
            models['lstm']['model'] = LSTMClassifier(vocab_size=len(models['lstm']['word2int']), embed_dim=64, hidden_dim=128, num_statuses=len(models['lstm']['status2int']))
            models['lstm']['model'].load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
            models['lstm']['model'].eval()

            # Define inference function
            def inference_lstm(texts):
                preds = []
                for text in texts:
                    tokens = re.findall(r'\w+', text.lower())[:150]
                    ids = [models['lstm']['word2int'].get(token, models['lstm']['word2int']['<UNK>']) for token in tokens]
                    ids += [models['lstm']['word2int']["<PAD>"]] * (150 - len(ids))
                    x = torch.tensor(ids).unsqueeze(0)
                    with torch.no_grad():
                        logits = models['lstm']['model'](x)
                        pred = torch.argmax(logits, dim=1).item()
                    preds.append(models['lstm']['int2status'][pred])
                return preds[0]
            models['lstm']['inference'] = inference_lstm
            print("LSTM loaded successfully")

        if os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_ENCODER_PATH):

            # Get checkpoint info
            checkpoint = torch.load(BERT_MODEL_PATH, map_location=DEVICE)

            # Initialize model: DistilBert + Classifier head
            models['bert']['model'] = DistilBertModel.from_pretrained(pretrained_model_name_or_path="distilbert-base-uncased")
            models['bert']['classifier'] = nn.Sequential(
                nn.Linear(768 + 4, 768), # Hidden size + len(numerical_columns)
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(768, checkpoint['classifier_state_dict']['3.weight'].shape[0]),
            )

            # Additional items necessary for the model
            models['bert']['label_encoder'] = joblib.load(BERT_ENCODER_PATH)
            models['bert']['tokenizer'] = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            # Load weights from the checkpoint
            models['bert']['model'].load_state_dict(checkpoint["bert_state_dict"])
            models['bert']['classifier'].load_state_dict(checkpoint['classifier_state_dict'])
            models['bert']['model'] = models['bert']['model'].to(DEVICE)
            models['bert']['classifier'] = models['bert']['classifier'].to(DEVICE)
            models['bert']['model'].eval()
            models['bert']['classifier'].eval()

            # Forward function definition
            def forward_bert(texts):
                encoding = models['bert']['tokenizer'](texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
                outputs = models['bert']['model'](**encoding)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = embeddings.to(DEVICE)
                numerical_features = []
                for text in texts:
                    first_person_ratio = helper.word_ratio(text, FIRST_PERSON)
                    negatives_ratio = helper.word_ratio(text, NEGATIVES)
                    suicide_ratio = helper.word_ratio(text, SUICIDE_WORDS)
                    repeat_ratio = helper.reps_ratio(text)
                    numerical_features.append([first_person_ratio, negatives_ratio, suicide_ratio, repeat_ratio])
                numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
                numerical_features = numerical_features.to(DEVICE)
                combined_features = torch.cat([embeddings, numerical_features], dim=1)
                combined_features = combined_features.to(DEVICE)
                logits = models['bert']['classifier'](combined_features)
                del embeddings
                del outputs
                del numerical_features
                del combined_features
                torch.cuda.empty_cache()
                return logits

            # Inference function definition
            def inference_bert(texts):
                with torch.no_grad():
                    logits = forward_bert(texts)
                pred = torch.argmax(logits, dim=1).item()
                return models['bert']['label_encoder'].inverse_transform([pred])[0]
            models['bert']['inference'] = inference_bert
            print("Finetuned BERT loaded successfully")

        if os.path.exists(BINARY_NN_PATH):

            # Initialize model
            models['binary_nn']['model'] = tf.keras.models.load_model(BINARY_NN_PATH)

            # Inference function definition
            def inference_nn(texts):
                preds = models['binary_nn']['model'].predict(tf.constant([texts], dtype=tf.string))
                label = "abnormal" if preds[0] >= 0.5 else "normal"
                return label
            models['binary_nn']['inference'] = inference_nn
            print("Binary NN loaded successfully")

    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# Database setup
DATABASE = 'journal.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS journal_entries (
                date TEXT PRIMARY KEY,
                content TEXT
            )
        ''')
        conn.commit()

init_db()

@app.route('/')
def index():
    return "Backend is up and running!"

@app.route('/api/journal', methods=['GET'])
def get_journal():
    date = request.args.get('date')
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    with get_db_connection() as conn:
        entry = conn.execute('SELECT content FROM journal_entries WHERE date = ?', (date,)).fetchone()

    if entry:
        return jsonify({"date": date, "content": entry['content']})
    else:
        return jsonify({"date": date, "content": ""})

@app.route('/api/journal', methods=['POST'])
def save_journal():
    data = request.json
    date = data.get('date')
    content = data.get('content')

    if not date:
        return jsonify({"error": "Date is required"}), 400

    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO journal_entries (date, content)
            VALUES (?, ?)
            ON CONFLICT(date) DO UPDATE SET content = excluded.content
        ''', (date, content))
        conn.commit()

    return jsonify({"status": "success"})

@app.route('/api/hello')
def hello():
    return {"message": "Hello from Flask!"}

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    multi_type = data.get('multi_model', 'xgboost')
    binary_type = data.get('binary_model', 'nn')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    model = models.get(multi_type)
    if model is None:
        return jsonify({"error": f"Model {multi_type} not loaded"}), 500

    # Send for prediction, models will handle preprocessing
    try:
        if binary_type == 'nn':
            prediction = models['binary_nn']['inference']([text])
            print("Binary model prediction:", prediction)
            if prediction == 'abnormal':
                if multi_type == 'bert':
                    prediction = models['bert']['inference']([text])
                elif multi_type == 'lstm':
                    prediction = models['lstm']['inference']([text])
                else: # Default model is xgboost
                    prediction = models['xgboost']['inference']([text])
                print("Multiclass", multi_type, "prediction:", prediction)
        else:
            prediction = "normal"
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
