# 🍔 Food Bill Predictor AI

An end-to-end Machine Learning solution designed to predict the total cost of food orders. This project focuses heavily on **Advanced Data Cleaning** and **Feature Engineering** to transform a "messy" real-world dataset into a high-performing predictive model.

---

## 📥 Clone This Repository

```bash
git clone https://github.com/Kaushik4636/Food-Bill-Predictor.git
cd Food-Bill-Predictor
```

---

## 🛠️ The "Messy Data" Challenge

The primary objective of this project was to handle a dataset containing inconsistent formatting, missing values, and high-variance outliers.

### Key Preprocessing Steps:

* **Data Standardization:** Used `.str.strip().str.capitalize()` to fix inconsistent categorical entries (e.g., "MALE", "male ", and "Male").
* **Intelligent Imputation:** Applied median imputation for numerical gaps and mode imputation for categorical gaps to maintain dataset integrity.
* **Outlier Mitigation:** Filtered extreme values using quantile analysis (top 5%) to prevent the Linear Regression model from being skewed by "whale" orders.
* **Multi-Collinearity Check:** Analyzed relationships between features like `num_items` and `delivery_distance` to ensure independent predictive power.

---

## 🔬 Technical Architecture

### 1. Model Development

* **Algorithm:** Multiple Linear Regression
* **Encoding:** One-Hot Encoding for `cuisine_type`, `meal_time`, and `weekend` status
* **Scaling:** Standardized features to a common scale ($Z$-score) to ensure the model correctly weights distance vs. item count

---

### 2. Performance Metrics

* **R² Score:** `0.82+` (Varies by training split)
* **Mean Absolute Error (MAE):** Optimized to minimize the average dollar-amount deviation per order

---

## 💻 Tech Stack

* **Language:** Python 3.11
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Deployment:** Streamlit (Glassmorphism UI)
* **Serialization:** Pickle (`.pkl`) for version-locked model deployment

---

## 📁 Repository Structure

```text
├── app.py                      # Streamlit UI & Preprocessing Logic
├── Food_bill_predictor.ipynb   # Jupyter Notebook (Model Development & EDA)
├── model_data.pkl              # Serialized Model & Scaling Parameters
├── food_messy_dataset.csv      # Original Raw Dataset
├── requirements.txt            # Production Dependencies
└── README.md                   # Documentation
```

---

## 👤 Author

**Kaushik**
*Data Science Intern | B.Tech IT*
