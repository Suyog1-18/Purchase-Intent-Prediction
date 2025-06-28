# 🧠 Customer Intention Predictor (0% - 100%)

This project predicts **customer purchase intention as a percentage (0–100%)** using a machine learning model built with **Gradient Boosting Regressor** in Python. It also includes SQL-based behavioral analysis to extract meaningful insights from user activity logs.

---

## 📁 Project Structure

**customer-intention-predictor**
─ project_ml.ipynb # ML model training and evaluation.
─ model.pkl # Trained ML model.
─ requirements.txt # Python dependencies.
─ data2.csv # Cleaned/preprocessed dataset.
─ coustumer_data_for_ml_2.csv # Raw input dataset.
─ MySQL_project.sql # SQL query for user behavior analysis.
─ README.md # Project documentation.

yaml
Copy
Edit

---

## 🚀 Objective

- Train a regression model to predict **purchase intention score (%)** based on user behavior data.
- Analyze customer interaction data (views, add-to-cart, transactions) using SQL.
- Deploy the model with a user-friendly Gradio interface.

---

## 📊 Model Details

- **Algorithm**: `GradientBoostingRegressor` (scikit-learn)
- **Target Output**: Continuous percentage (0% to 100%)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

---

# Example input: "0.5, 12, 3"
Predicted Customer Intention: 83.45%
🧮 SQL Component
File: MySQL_project.sql

This query performs customer session analysis:

Total views, add-to-cart, and transactions per visitorid

Session duration in minutes

sql
Copy
Edit
SELECT 
    visitorid,
    COUNT(CASE WHEN event = 'view' THEN 1 END) AS no_views,
    COUNT(CASE WHEN event = 'addtocart' THEN 1 END) AS no_addtocart,
    COUNT(CASE WHEN event = 'transaction' THEN 1 END) AS no_transaction,
    (MAX(timestamp) - MIN(timestamp)) / (1000 * 60) AS time_spent_minutes
FROM events
GROUP BY visitorid
ORDER BY visitorid;
This analysis was used to prepare features for the machine learning model.

📂 Data Files
coustumer_data_for_ml_2.csv – Raw data from customer session logs.

data2.csv – Cleaned and feature-engineered dataset used for model training.

💡 Future Improvements
Deploy app using Hugging Face Spaces or Streamlit Cloud

Incorporate time-based features and device/user type

Add SHAP values for model explainability

Auto-sync SQL → ML pipeline

👨‍💻 Author
Suyog Bewle
B.Tech, Mechanical Engineering
Indian Institute of Technology, Ropar
LinkedIn (Add your profile)

