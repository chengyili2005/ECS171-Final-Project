To run the application:

1.  Navigate to the project root directory.
2.  Run the Flask backend:
    ```bash
    uv run python app.py
    ```
    (This will serve the React build and the API. It runs in debug mode, so it will automatically reload on code changes.)
3.  Open your browser to `http://127.0.0.1:5000/`.

For frontend development (with hot reloading):

1.  Start the Flask backend in one terminal:
    ```bash
    uv run python app.py
    ```
2.  In a separate terminal, navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
3.  Start the React development server:
    ```bash
    npm start
    ```
    (This will usually open on `http://localhost:3000`. You might need to add a proxy to `frontend/package.json` if you encounter API communication issues, like `"proxy": "http://127.0.0.1:5000"`.)