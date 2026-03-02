from flask import Flask, send_from_directory, request, jsonify
import os
import sqlite3
import joblib # For loading XGBoost/Scikit-learn models
# import tensorflow as tf # Uncomment if using Keras/TF for NN
# import torch # Uncomment if using PyTorch for NN

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

# Model paths
MODELS_DIR = 'models'
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_multiclass.joblib')
BINARY_NN_PATH = os.path.join(MODELS_DIR, 'binary_nn.h5') # or .pth

# Load models globally (placeholders - will fail until files exist)
models = {
    'xgboost': None,
    'binary_nn': None
}

def load_models():
    try:
        if os.path.exists(XGB_MODEL_PATH):
            models['xgboost'] = joblib.load(XGB_MODEL_PATH)
        if os.path.exists(BINARY_NN_PATH):
            # Example for Keras: models['binary_nn'] = tf.keras.models.load_model(BINARY_NN_PATH)
            pass
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
    model_type = data.get('model_type', 'xgboost') # Default to xgboost

    if not text:
        return jsonify({"error": "No text provided"}), 400

    model = models.get(model_type)
    if model is None:
        return jsonify({"error": f"Model {model_type} not loaded"}), 500

    # Placeholder logic for prediction
    # In reality, you'd need to preprocess 'text' first (e.g., tokenization, vectorization)
    try:
        if model_type == 'xgboost':
            # prediction = model.predict(processed_text)
            prediction = "XGBoost result placeholder"
        else:
            # prediction = model.predict(processed_text)
            prediction = "NN result placeholder"
            
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
