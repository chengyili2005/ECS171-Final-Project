import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');

  const handlePredict = () => {
    console.log('Predicting for:', inputText);
    // Future: Add backend call here
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
          <button onClick={handlePredict}>Predict</button>
        </div>
      </header>
    </div>
  );
}

export default App;
