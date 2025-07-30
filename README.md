# 🔍 ANN Classification for Bank Churn Prediction

This project aims to predict whether a customer will **churn** (leave the bank) using an **Artificial Neural Network (ANN)**. The model is trained on a publicly available bank customer dataset and deployed via **Streamlit** for interactive usage.

🚀 **Live Demo**: [Open App](https://ann-churn-classification-wfae88vfd5u4r4vhlac6oa.streamlit.app/)

---

## 📌 Project Overview

Customer churn is a key metric for businesses, especially in the banking sector. This project:
- Uses an ANN for binary classification (churn vs. not churn).
- Takes various customer features as input (age, balance, credit score, etc.).
- Predicts the likelihood of a customer leaving the bank.

---

## 🧠 Model Architecture

The ANN model was built using **TensorFlow/Keras** and has the following structure:

- Input Layer: 11 features (after preprocessing)
- Hidden Layer 1: Dense layer with ReLU activation
- Hidden Layer 2: Dense layer with ReLU activation
- Output Layer: Single neuron with Sigmoid activation

---

## 📊 Dataset

- **Source**: Kaggle or other open-source bank churn datasets.
- **Target Variable**: `Exited` (1 if customer churned, 0 otherwise)
- **Features Include**:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **ML Libraries**: TensorFlow, Scikit-learn, Pandas, NumPy

---

## ⚙️ How It Works

1. User enters customer details via the Streamlit UI.
2. Inputs are preprocessed (label encoding, scaling).
3. The trained ANN model predicts the churn probability.
4. Output is displayed: `Will Churn` or `Will Not Churn`.

---

## 🚀 Getting Started (Local Setup)

```bash
# Clone the repo
git clone https://github.com/yourusername/ann-churn-prediction.git
cd ann-churn-prediction

# Create virtual environment & activate it
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
