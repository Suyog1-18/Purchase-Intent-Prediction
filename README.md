# 🧠 Customer Intention Predictor (0% - 100%)

This project predicts **customer purchase intention** as a percentage between 0% and 100% using a regression model trained on customer interaction data. It also includes SQL-based behavioral feature extraction and a Gradio-based web interface for live model interaction.

---

## 📁 Project Structure
- A Jupyter Notebook with the complete ML workflow (project_ml.ipynb)

- SQL query used for feature engineering (MySQL_project.sql)

- Cleaned datasets (coustumer_data_for_ml_2.csv, data2.csv)

- Trained model (model.pkl)

- Python dependencies (requirements.txt)



---

## 🚀 Objective

- Predict how likely a customer is to make a purchase based on behavioral data.
- Build a regression model that outputs the **purchase intention** as a percentage.
- Analyze raw event logs using SQL and use the output as features for ML.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn** for machine learning
- **joblib** for model saving/loading
- **MySQL** for SQL-based behavioral feature extraction
- **Jupyter Notebook** for development and EDA

---

## 📊 Model Information

- **Algorithm**: GradientBoostingRegressor
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
