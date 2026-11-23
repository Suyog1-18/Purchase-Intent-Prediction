# 🛒 Customer Purchase Intent Prediction

Predict whether an e-commerce visitor will make a purchase **during their current session**, based on their browsing behavior. The project covers the full ML workflow: session building from raw clickstream data, exploratory analysis, model comparison, hyperparameter tuning, threshold selection, and a **Streamlit** web app for live predictions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Key EDA Insights](#-key-eda-insights)
- [Model Results](#-model-results)
- [Running the App](#-running-the-app)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Overview

Most visitors to an online store never buy anything. If a store can identify, *while the visitor is still browsing*, who is likely to purchase, it can act on it: show a discount, trigger a chat prompt, or avoid wasting marketing spend on low-intent sessions.

This project turns raw event logs (views, add-to-carts, transactions) into **session-level features** and trains a classifier to output the **probability that a session ends in a purchase**.

**Challenge:** the data is extremely imbalanced. Only about **0.81%** of sessions contain a purchase, so accuracy alone is misleading and the project focuses on **PR-AUC, precision, recall and F1**.

---

## 📊 Dataset

Raw e-commerce clickstream events (`events.csv`), the RetailRocket-style format:

| Column | Description |
|---|---|
| `timestamp` | Event time (Unix ms) |
| `visitorid` | Unique visitor ID |
| `event` | `view`, `addtocart` or `transaction` |
| `itemid` | Item the event relates to |
| `transactionid` | Present only for purchases |

- **2,756,101** raw events from **1,407,580** visitors
- Converted into **1,761,675 sessions**
- **14,297** purchase sessions vs **1,747,378** non-purchase sessions

> The raw dataset is not included in this repo because of its size. Place `events.csv` in the project root to re-run the notebooks.

---

## 📁 Project Structure

```
├── data_prepration.ipynb          # Sessionization + feature engineering
├── Project_EDA.ipynb              # Exploratory data analysis
├── Project_model_bulding.ipynb    # Model training, comparison, tuning, saving
├── app.py                         # Streamlit app for live predictions
├── purchase_intent_model.pkl      # Final tuned XGBoost model
├── threshold.pkl                  # Decision threshold used by the app
└── README.md
```

---

## 🔬 Methodology

### 1. Data preparation (`data_prepration.ipynb`)
- Converted timestamps to datetime and sorted events per visitor.
- **Sessionization:** a new session starts when a visitor has been inactive for **more than 30 minutes**. A session can be longer than 30 minutes as long as the gaps between events stay under 30 minutes.
- Aggregated events into one row per session.
- **Target:** `1` if the session has at least one `transaction`, otherwise `0`.

**Features used (7):**

| Feature | Description |
|---|---|
| `views` | Number of item views in the session |
| `add_to_cart` | Number of add-to-cart events |
| `unique_item_viewed` | Number of distinct items viewed |
| `session_duration_min` | Time between first and last event (minutes) |
| `month` | Month the session started |
| `day` | Day of week (0 = Monday … 6 = Sunday) |
| `hour` | Hour of day (0–23) |

### 2. Exploratory analysis (`Project_EDA.ipynb`)
Class balance, feature distributions, conversion rate by cart activity, session duration, hour and weekday, and a correlation heatmap.

### 3. Modeling (`Project_model_bulding.ipynb`)
- Stratified 80/20 train/test split (`random_state=42`) to preserve the class ratio.
- Trained and compared **four models**, all handling imbalance via class or sample weights:
  Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
- Selected the best candidates using **PR-AUC**, then ran a threshold sweep (0.05 to 0.99) to look at the precision/recall/F1 trade-off.
- Tuned **XGBoost** with `RandomizedSearchCV` (15 candidates, 3-fold CV, scored on PR-AUC).
- Saved the final model and threshold with `joblib`.

---

## 💡 Key EDA Insights

- **Add-to-cart is the strongest signal.** Sessions without an add-to-cart convert at only ~**0.13%**, versus ~**27%** for sessions with one.
- **Purchasers browse more.** Average session of a buyer: ~6.2 views, ~1.5 add-to-carts, ~28.8 minutes. Non-buyers: ~1.5 views, ~0.03 add-to-carts, ~1.5 minutes.
- **Longer sessions convert more.** Purchase rate rises as session duration increases.
- **Timing matters a little.** Purchase rate peaks around **3 PM** and on **Wednesdays**.
- **Skewed data.** The median session has a single view and zero duration.

---

## 🏆 Model Results

Test set of 352,335 sessions (2,859 purchases), default 0.5 threshold:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.149 | 0.859 | 0.254 | 0.978 | 0.331 |
| Random Forest | 0.541 | 0.774 | 0.637 | 0.988 | 0.674 |
| Gradient Boosting | 0.189 | 0.970 | 0.316 | 0.994 | 0.625 |
| XGBoost | 0.211 | 0.966 | 0.347 | 0.995 | 0.697 |
| **Tuned XGBoost (final)** | **0.713** | **0.491** | **0.582** | **0.995** | **0.712** |

**Final model:** tuned XGBoost, which has the best PR-AUC (0.712) and ROC-AUC (0.995). Its best hyperparameters:
`n_estimators=300`, `learning_rate=0.1`, `max_depth=3`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.7`, `gamma=0.5`, `scale_pos_weight=1`.

**Feature importance:** `add_to_cart` dominates (~71%), followed by session duration (~11%), unique items viewed (~10%) and views (~7%). Month, day and hour contribute very little.

---

## 🚀 Running the App

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install streamlit pandas scikit-learn xgboost joblib
```

### 3. Check the model paths in `app.py`
The app currently loads the files from a local Desktop path. Change it to relative paths so it works anywhere:

```python
model = joblib.load("purchase_intent_model.pkl")
threshold = joblib.load("threshold.pkl")
```

### 4. Launch
```bash
streamlit run app.py
```

### Using the app
Enter a session's behavior (views, add-to-cart count, unique items viewed, duration, month, day of week, hour) and click **Predict Purchase Intent**. The app shows the **purchase probability** and labels the session:

- 🟢 **Likely to Purchase** if probability ≥ threshold
- 🔵 **Unlikely to Purchase** otherwise

---

## 🧰 Tech Stack

- **Python 3.12**
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, XGBoost
- **Deployment:** Streamlit
- **Persistence:** joblib

---

## ⚠️ Limitations & Future Work

- **Threshold:** the saved threshold (`0.981`) was picked during the threshold sweep on the *untuned* XGBoost model. The final tuned model was trained with different class weighting, so its probabilities are distributed differently. Re-running the threshold sweep on the tuned model is recommended.
- **Few features:** the model uses only session-level counts and time features. Ideas for improvement: item/category information, visitor history (returning vs new), time between events, and price data.
- **Probability calibration:** class weighting inflates probabilities, so outputs should be read as a ranking score rather than a true likelihood unless calibrated.
- **Single dataset / time window:** performance may differ on other stores or seasons.
- **Next steps:** SHAP explanations, calibration, cost-based threshold selection, and deploying the app (Streamlit Community Cloud / Docker).

---

## 👤 Author
**<Suyog Bewle>**

---

⭐ If you found this project useful, consider giving it a star!
