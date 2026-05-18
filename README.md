# Stroke Prediction Analysis (Mini-Project)

A focused mini-project demonstrating foundational data preprocessing, mathematical modeling, and exploratory data analysis (EDA) on the **Stroke Prediction Dataset**.

This project demonstrates practical applications of:
- NumPy vectorized computation
- Mathematical optimization (Gradient Descent)
- Probability & Linear Algebra
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data preprocessing for Machine Learning

**Author:** Trần Gia Hy 

---

# 📌 Project Overview

According to the World Health Organization (WHO), stroke is the **2nd leading cause of death globally**, accounting for approximately **11% of total deaths** worldwide.

This project implements a structured and production-style workflow to:
- Clean and preprocess healthcare datasets
- Perform statistical and mathematical analysis
- Engineer meaningful healthcare features (e.g., Glucose to BMI ratio)
- Visualize hidden data patterns through an automated **2x2 Executive Dashboard**
- Optimize regression parameters using Gradient Descent
- Prepare a machine-learning-ready dataset

---

# 📂 Project Structure

```text
stroke-prediction-project/
├── data/
│   ├── raw/
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/
│       └── clean_stroke_data.csv
│
├── notebooks/
│   └── analysis.ipynb               # Interactive notebook & Master Dashboard generator
│
├── src/
│   ├── numpy_tasks.py               # Statistical & normalization tasks
│   ├── math_tasks.py                # Gradient Descent & Probability tasks
│   ├── pandas_tasks.py              # Data cleaning, EDA & encoding tasks
│   └── utils.py                     # Data loader and path config
│
├── outputs/
│   └── figures/
│       ├── Stroke_Analysis_Summary.png    # 2x2 Master Dashboard (All-in-one)
│       ├── gradient_descent_convergence.png           # Math Module: MSE Loss curve
│       ├── Age_stroke_dist.png            # Pandas Q5: Histogram & Countplot
│       ├── Correlation_heatmap.png        # Pandas Q9: Pure Numeric Heatmap
│       └── Glucose_boxplot.png            # Pandas Q9: Target variable analysis
│
├── report/
│   └── Final_report.pdf             # Detailed analytical findings and insights
│
├── requirements.txt
└── README.md
```

## 🛠️ Module Breakdown

### 📘 Module 1 — NumPy Advanced Manipulations
#### ✅ Data Loading & Cleaning
- Extract numerical healthcare features: `age`, `avg_glucose_level`, `bmi`.
- Handle missing values using column-wise mean imputation.

#### ✅ Statistical Profiling
- Implemented statistical calculations manually: Mean, Median, Standard Deviation.

#### ✅ Feature Scaling
- Custom vectorized implementations of Min-Max Scaling.

---

### 📗 Module 2 — Mathematics for AI
#### ✅ Probability & Risk Estimation
- Computed Prior probability: $P(\text{stroke}=1)$
- Computed Conditional probability: $P(\text{stroke}=1 \mid \text{hypertension}=1)$

#### ✅ Gradient Descent Optimization
- Implemented Linear Regression from scratch using Mean Squared Error (MSE), partial derivatives, and Iterative Gradient Descent updates.

---

### 📙 Module 3 — Pandas & Exploratory Data Analysis
#### ✅ Data Transformation
- Removed low-frequency anomalies (e.g., `"Other"` gender entries).
- Applied One-Hot Encoding using `drop_first=True` to prevent multicollinearity.

#### ✅ Feature Engineering
- Created healthcare-derived features such as `glucose_to_bmi_ratio`.
- Generated demographic categories: Age Groups & BMI Categories.

#### ✅ Advanced Visualization (Bug-Free)
- Engineered a robust pipeline to bypass Seaborn formatting bugs, generating high-resolution numerical heatmaps and distribution charts.

---

## 📊 Visual Analytics & Master Dashboard

A key highlight of this project is the **Stroke Analysis Summary Dashboard**, generated dynamically via `analysis.ipynb`. It aggregates the outputs of all three modules into a single 2x2 high-resolution grid for executive reporting.

Generated visualization outputs include:

| Visualization | Description |
| :--- | :--- |
| `Stroke_Analysis_Summary.png` | **Master Dashboard 2x2** compiling all project visual insights |
| `gradient_descent_convergence.png` | Gradient Descent convergence (MSE Train & Test) |
| `Age_stroke_dist.png` | Stroke distribution by age and class imbalance |
| `Correlation_heatmap.png` | Correlation between numeric variables |
| `Glucose_boxplot.png` | Glucose distribution analysis across stroke status |

---

## 🚀 Getting Started

### ✅ Prerequisites
Ensure you have Python 3.8+ and the `pip` package manager installed.

Install dependencies:
```bash
pip install -r requirements.txt
```
---

## ▶️ Execution Pipeline

Run each module independently from the project root directory.

### Run Module 1 — NumPy Operations
```bash
python src/numpy_tasks.py
```

Run Module 2 — Mathematics & Optimization
```bash
python src/math_tasks.py
```

Run Module 3 — Pandas EDA & Visualization
```bash
python src/pandas_tasks.py
```

---

## 📓 Jupyter Notebook
To explore the interactive notebook and generate the Master Dashboard:
```bash
jupyter notebook notebooks/analysis.ipynb
```
The notebook acts as the presentation layer, dynamically importing the src modules to compute logic, run algorithms, and stitch together the final 2x2 analytical framework.

--- 

## 📈 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Mathematical Foundations (Linear Algebra, Calculus, Probability)
- Development Tools (Jupyter, Git, VS Code)
