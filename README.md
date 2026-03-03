# ECS171-Final-Project
Sentiment Analysis for mental health guardrails

# Information
TODO

# Setup

Clone this directory
```
git clone https://github.com/chengyili2005/ECS171-Final-Project.git
```

Download our models and import them into the `models/` directory
1. Go to our [Google Drive](https://drive.google.com/drive/folders/1cI5vjc2AmUdvT4I3CQk7DZd5GbJiVfvd?usp=sharing) and download the models
2. Drag them into `models/`
   - `models/` should look like:
   ```
   - models/
    - bert_classifier.py
    - bert_label_encoder.joblib
    - lstm_classifier.pt
    - lstm_status2int_mapping.json
    - lstm_word2int_mapping.json
    - neural_network_classifier.keras
   - ...
   ```

Install general dependencies
- Ensure Python is installed
  ```bash
  sudo apt install python3 # Ubuntu
  ```
  ```bash
  brew install python # macOS (I think)
  ```
- Ensure NPM (for frontend) is installed
  ```bash
  sudo apt install nodejs npm # Ubuntu
  ```
  ```bash
  brew install node # macOS (I think)
  ```
- Ensure UV (for backend) is installed
  ```bash
  snap install astral-uv # Ubuntu
  ```
  ```bash
  brew install uv # macOS (I think)
  ```

Setting up the backend
- Install backend dependencies
  ```bash
  # Go to backend directory
  cd 171_app/

  # Sync dependencies
  uv sync
  ```

- Start backend server
  ```bash
  # Still in ECS-171-Final-Project/171_app
  uv run python app.py
  ```

- Check if backend is up
  - Open your browser to `http://127.0.0.1:5000/` to ensure the URL exists.

Setting up the frontend
- Install frontend dependencies
  ```bash
  # Assuming you are in the ECS-171-Final-Project directory
  cd 171_app/frontend

  # Install node dependencies
  npm install
  ```
  - Debugging: I had a problem with react-scripts, but the following fixed it
    ```bash
      npm install react-scripts@5.0.1 --save # If not already installed
      echo "DANGEROUSLY_DISABLE_HOST_CHECK=true" > .env
    ```
- Start the frontend
  ```bash
  # Still in the 171_app/frontend directory
  npm start
  ```
  - Debugging: A note from `171_app/RUN_INSTRUCTIONS.md`
      - (This will usually open on `http://localhost:3000`. You might need to add a proxy to `frontend/package.json` if you encounter API communication issues, like `"proxy": "http://127.0.0.1:5000"`.)

The frontend is now being displayed in `http://localhost:3000`, and the backend `http://127.0.0.1:5000` is being called via the frontend.

