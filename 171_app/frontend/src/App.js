import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [Model, setModel] = useState('xgboost');

  const handlePredict = async () => {
    console.log('Predicting for:', inputText);
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, binary_model: 'nn', multi_model : Model }),
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data.prediction);
      } else {
        setResult(`Error: ${data.error}`);
      }
    } catch (err) {
      setResult(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Predictor</h1>
        <div className="container">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text here..."
            rows="10"
            cols="50"
          />
          <br />

          <div style={{ marginTop: '16px' }}>
            <label htmlFor="select-model" style={{ marginRight: '10px' }}>
              Multiclass Model:
            </label>
            <select
              id="select_model"
              value={Model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="xgboost">XGBoost (simple)</option>
              <option value="lstm">LSTM (medium)</option>
              <option value="bert">BERT (complex)</option>
            </select>
          </div>

          <button onClick={handlePredict} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>

          {result && (
            <div style={{ margintop: '20px', padding: '12px', background: '#1e1e1e', borderRadius: '8px'}}>
              <strong></strong> {result}
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
