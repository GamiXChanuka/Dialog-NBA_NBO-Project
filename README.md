# Dialog Smart NBA & NBO Predictor

An AI-powered Next Best Action (NBA) and Next Best Offer (NBO) recommendation engine designed to optimize customer engagement, reduce churn, and hyper-personalize telecom services.

<img width="1523" height="1290" alt="image" src="https://github.com/user-attachments/assets/9e27570b-4c38-4ddc-816a-8cb19c2fd9ab" />


## 📌 The Problem It Solves

Modern telecom operators deal with millions of customers, each with unique data habits, spending patterns, and device capabilities. A "one-size-fits-all" marketing approach leads to:
1. **High Churn Rates**: Customers leaving because their plans no longer fit their needs.
2. **Missed Revenue**: Failing to upsell data packages to customers who regularly exceed their limits.
3. **Irrelevant Offers**: Annoying customers with SMS campaigns for offers they will never accept (e.g., offering a 5G data pack to a user with a 3G basic phone).

**The Solution:**
This project uses Machine Learning (Random Forest Classifier) to analyze 19 different data points per customer (including data usage, complaints, reload frequency, and current device). It then predicts the **Next Best Action** (the strategy to use) and the **Next Best Offer** (the specific product to pitch), along with the AI's confidence percentage for that decision.

---

## 🚀 How to Run the Project

This project consists of two main parts: a Python FastAPI backend and a React/Vite frontend.

### 1. Start the Backend API (Python)
The backend serves the ML model predictions via a REST API.

1. Open your terminal and navigate to the project directory.
2. Ensure you have the required dependencies installed:
   ```bash
   cd server
   pip install fastapi uvicorn pydantic scikit-learn pandas joblib
   ```
   *(Note: You can also use `pip install -r requirements.txt` if available).*
3. Run the FastAPI server:
   ```bash
   python app.py
   ```
4. The API will start running on **`http://127.0.0.1:8000`**.
   *(You can view the interactive Swagger API documentation by navigating to `http://127.0.0.1:8000/docs` in your browser).*

### 2. Start the Frontend UI (React)
The frontend is a modern, glassmorphism web interface where agents can input customer details and generate instant AI insights.

1. Open a **new** terminal window and navigate to the `www` directory:
   ```bash
   cd www
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser and go to **`http://localhost:5173`**.

---

## 🧠 Project Structure

- **`ML model/`**: Contains the Random Forest training script (`main.py`), the dataset, and the serialized `.pkl` models. 
- **`server/`**: Contains the FastAPI application (`app.py`) that loads the `.pkl` files and provides the `/predict` endpoint.
- **`www/`**: Contains the React + Vite frontend application. It talks directly to the local Python server to fetch predictions.

## 🛠 Features

- **Dynamic Data Parsing**: Frontend intelligently handles dropdown states and parses data structures specific to what the ML model requires.
- **Confidence Scoring**: Not only tells you *what* to do, but exactly *how confident* the AI is in its recommendation.
- **Top Alternatives**: Provides the 2nd and 3rd best fallback actions in case the customer rejects the primary offer. 
- **Beautiful UI**: Modern glassmorphism design language using Dialog's branding aesthetics.
