# Iris Species Classification API

## Description
This API serves a Machine Learning model trained to classify Iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica**. It utilizes a **Random Forest Classifier** trained on the classic Iris dataset, accepting four physical measurements (sepal length, sepal width, petal length, and petal width) as input and returning the predicted species along with probability scores.

## How to Run
Follow these steps to get the API running on your local machine:

1.  **Navigate to the project folder:**
    ```bash
    cd deployment
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install flask joblib numpy scikit-learn
    ```
3.  **Start the Flask server:**
    ```bash
    python app.py
    ```
    *Note: The server will be active at `http://127.0.0.1:5000`.*

---

## API Specification

### 1. Health Check
Checks if the API is running correctly.
* **URL:** `/health`
* **Method:** `GET`
* **Response:** `{"status": "healthy"}`

### 2. Single Prediction
Predicts the species for a single iris flower.
* **URL:** `/predict`
* **Method:** `POST`
* **Request Body (JSON):**
    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```
* **Response Format (JSON):**
    ```json
    {
      "predicted_class": "setosa",
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      }
    }
    ```

### 3. Batch Prediction
Predicts species for multiple samples at once.
* **URL:** `/predict_batch`
* **Method:** `POST`
* **Request Body (JSON):**
    ```json
    {
      "samples": [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.1, 4.4, 1.4]
      ]
    }
    ```

---

## Example Usage (Python)
You can test the API using the `requests` library:

```python
import requests

url = "[http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)"
payload = {"features": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(url, json=payload)
print(response.json())