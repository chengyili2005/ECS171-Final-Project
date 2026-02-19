from flask import Flask, send_from_directory, request, jsonify
import os
import sqlite3

app = Flask(__name__, static_folder='frontend/build', static_url_path='')

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


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
