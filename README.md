<div align="center">

# 🧠 Psyche Risk
## An Explainable End-to-End Machine Learning Pipeline for Multiclass Mental Health Risk Assessment and Stratification Using Lifestyle, Stress & Clinical Features

*Predict. Understand. Protect.*

<br/>

[![🚀 Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://psyche-risk-zjb4qjvfx6dg2ojv6pyft2.streamlit.app/)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/196JQASaDTK8a7VoDUmnavN-LUFWjYUBk)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-2E8B57?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> An end-to-end machine learning pipeline that classifies an individual's mental health risk level —
> **🟢 Low**, **🟡 Medium**, or **🔴 High** — from lifestyle, psychological, and demographic data.

<br/>

---

</div>

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🚀 Live Demo](#-live-demo)
- [📁 Repository Structure](#-repository-structure)
- [🔬 Dataset & Features](#-dataset--features)
- [⚙️ Feature Engineering](#️-feature-engineering)
- [🤖 Model & Training](#-model--training)
- [⚡ Quickstart](#-quickstart)
- [🧰 Tech Stack](#-tech-stack)
- [📊 Evaluation](#-evaluation)
- [⚠️ Disclaimer](#️-disclaimer)
- [👨‍💻 Author](#-author)

---

## ✨ Overview

**Psyche Risk** is a production-ready ML system that leverages gradient boosting and explainable AI to surface mental health risk from everyday behavioural signals.

<div align="center">

| 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk |
|:-----------:|:--------------:|:------------:|
| Class `0`   | Class `1`      | Class `2`    |

</div>

**What this project does:**

- 🔍 **Explores** mental health survey data with rich EDA (distributions, correlations, boxplots)
- 🛠️ **Engineers** four composite features that meaningfully compress raw psychological signals
- ⚡ **Trains** a tuned LightGBM classifier inside a full scikit-learn pipeline
- 🔬 **Optimises** hyperparameters with Optuna (601 estimators, Macro F1 objective)
- 🧩 **Explains** every prediction using SHAP TreeExplainer (beeswarm + summary plots)
- 🌐 **Deploys** an interactive Streamlit web application — live and publicly accessible

---

## 🚀 Live Demo

<div align="center">

### 👉 [psyche-risk-zjb4qjvfx6dg2ojv6pyft2.streamlit.app](https://psyche-risk-zjb4qjvfx6dg2ojv6pyft2.streamlit.app/)

[![Try the App](https://img.shields.io/badge/Click%20to%20Open%20Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://psyche-risk-zjb4qjvfx6dg2ojv6pyft2.streamlit.app/)

*Enter lifestyle and psychological details → receive an instant mental health risk classification.*

</div>

---

## 📁 Repository Structure

```
📦 psyche-risk/
│
├── 📓 Mentalrisk.ipynb               ← Full training, EDA & SHAP analysis notebook
├── 🤖 psyche_risk_model_full.joblib  ← Serialised LightGBM pipeline (preprocessor + model)
├── 📋 feature_names.json             ← Ordered list of 34 model input features
├── ⚙️  best_params.json              ← Optuna best hyperparameters (for reproducibility)
├── 🧪 test_sample.json               ← Sample JSON records for inference testing
└── 📄 README.md
```

---

## 🔬 Dataset & Features

The model consumes **34 features** across five categories:

<details>
<summary><b>👤 Demographic Features (5)</b></summary>
<br/>

| Feature | Type | Values |
|---|---|---|
| `age` | Numeric | Continuous |
| `gender` | Categorical | Male / Female / Other |
| `marital_status` | Categorical | Single / Married / Divorced |
| `education_level` | Categorical | High School / Bachelor / Master / PhD |
| `employment_status` | Categorical | Employed / Self-Employed / Student / Unemployed |

</details>

<details>
<summary><b>😴 Lifestyle Features (4)</b></summary>
<br/>

| Feature | Description |
|---|---|
| `sleep_hours` | Average nightly sleep duration |
| `physical_activity_hours_per_week` | Weekly exercise hours |
| `screen_time_hours_per_day` | Daily screen exposure |
| `working_hours_per_week` | Weekly work hours |

</details>

<details>
<summary><b>🧪 Psychological Scores — 1 to 10 scale (10)</b></summary>
<br/>

| Feature | Description |
|---|---|
| `anxiety_score` | Self-reported anxiety level |
| `depression_score` | Self-reported depression level |
| `stress_level` | General perceived stress |
| `mood_swings_frequency` | Frequency of mood fluctuations |
| `concentration_difficulty_level` | Difficulty focusing |
| `work_stress_level` | Workplace stress |
| `academic_pressure_level` | Academic burden |
| `financial_stress_level` | Financial anxiety |
| `social_support_score` | Perceived social support |
| `job_satisfaction_score` | Work satisfaction |

</details>

<details>
<summary><b>🏥 Clinical History — Binary (5)</b></summary>
<br/>

| Feature | Description |
|---|---|
| `panic_attack_history` | Past panic attacks (0/1) |
| `family_history_mental_illness` | Family psychiatric history (0/1) |
| `previous_mental_health_diagnosis` | Prior diagnosis (0/1) |
| `therapy_history` | Has received therapy (0/1) |
| `substance_use` | Substance use (0/1) |

</details>

<details>
<summary><b>⚙️ Engineered Features (4)</b></summary>
<br/>

| Feature | Description | Direction |
|---|---|---|
| `negative_load` | Mean of 8 stress & mood indicators | ↑ worse |
| `protective_factors` | Mean of social support, job satisfaction, physical activity | ↑ better |
| `sleep_quality` | Deviation from healthy sleep window (7–9 hrs) | ↓ worse |
| `work_life_imbalance` | Ratio of working hours to total waking hours | ↑ worse |

</details>

---

## 🤖 Model & Training

### Pipeline Architecture

```
Raw Input
    │
    ▼
┌─────────────────────────────────────────┐
│         ColumnTransformer               │
│  ├── Numerical → MedianImputer          │
│  │               + StandardScaler       │
│  ├── Categorical → OneHotEncoder        │
│  └── Binary → Pass-through             │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           LGBMClassifier                │
│   class_weight = 'balanced'             │
│   Tuned via Optuna (Macro F1)           │
└─────────────────────────────────────────┘
    │
    ▼
  🟢 Low   🟡 Medium   🔴 High
```

### Best Hyperparameters (via Optuna)

```json
{
  "n_estimators":       601,
  "learning_rate":      0.1911,
  "max_depth":          8,
  "num_leaves":         21,
  "min_child_samples":  73,
  "subsample":          0.6698,
  "colsample_bytree":   0.6552,
  "reg_alpha":          1.5782,
  "reg_lambda":         2.1119
}
```

### Training Configuration

| Setting | Value |
|---|---|
| Train / Test Split | 80% / 20% (stratified) |
| Cross-Validation | 3-fold (during Optuna search) |
| Optimisation Metric | Macro F1-score |
| Class Balancing | `class_weight='balanced'` |
| Explainability | SHAP TreeExplainer |
| Random Seed | `42` |

---

## ⚡ Quickstart

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/psyche-risk.git
cd psyche-risk
```

### 2 — Install dependencies

```bash
pip install pandas scikit-learn lightgbm optuna shap joblib streamlit
```

### 3 — Run inference

```python
import joblib, json
import pandas as pd

# Load saved pipeline
model = joblib.load("psyche_risk_model_full.joblib")

# Load sample records
with open("test_sample.json") as f:
    samples = json.load(f)

df = pd.DataFrame(samples)
predictions = model.predict(df)

label_map = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {label_map[pred]}")
```

### 4 — Launch the web app locally

```bash
streamlit run app.py
```

---

## 🧰 Tech Stack

<div align="center">

| Layer | Tool | Role |
|:---|:---|:---|
| Data | `pandas` · `numpy` | Ingestion & transformation |
| ML Pipeline | `scikit-learn` | Preprocessing, pipelines, evaluation |
| Model | `lightgbm` | Gradient boosting classifier |
| Tuning | `optuna` | Bayesian hyperparameter optimisation |
| Explainability | `shap` | SHAP TreeExplainer (feature attribution) |
| Serialisation | `joblib` | Model persistence |
| Visualisation | `matplotlib` · `seaborn` | EDA plots & confusion matrices |
| Deployment | `streamlit` | Interactive web application |
| Environment | Google Colab | Cloud training notebook |

</div>

---

## 📊 Evaluation

Model performance is assessed using a comprehensive suite:

- ✅ **Classification Report** — Per-class Precision, Recall, F1-score
- ✅ **Macro F1-score** — Primary metric (robust to class imbalance)
- ✅ **Confusion Matrix** — Heatmap visualisation across all three classes
- ✅ **SHAP Beeswarm & Summary Plots** — Global feature importance per risk class
- ✅ **Cross-validation scores** — Stability check during Optuna search

---

## ⚠️ Disclaimer

> This project is intended **strictly for educational and research purposes**.
> It is **not** a clinical diagnostic tool and must **not** replace professional mental health evaluation or treatment.
> If you or someone you know is experiencing mental health difficulties, please seek guidance from a qualified healthcare professional.

---

## 👨‍💻 Author

<br/>

<div align="center">

## **Agha Wafa Abbas**

*Lecturer in Computing | Machine Learning Researcher*

<br/>

| 🏛️ Institution | 📍 Location |
|:---|:---|
| 🎓 School of Computing, **University of Portsmouth** | Winston Churchill Ave, Southsea, Portsmouth PO1 2UP, UK |
| 🎓 School of Computing, **Arden University** | Coventry, United Kingdom |
| 🎓 School of Computing, **Pearson** | London, United Kingdom |
| 🎓 School of Computing, **IVY College of Management Sciences** | Lahore, Pakistan |

<br/>

### 📧 Contact

[![Portsmouth](https://img.shields.io/badge/agha.wafa%40port.ac.uk-University%20of%20Portsmouth-003087?style=for-the-badge&logo=microsoftoutlook&logoColor=white)](mailto:agha.wafa@port.ac.uk)

[![Arden](https://img.shields.io/badge/awabbas%40arden.ac.uk-Arden%20University-8B0000?style=for-the-badge&logo=microsoftoutlook&logoColor=white)](mailto:awabbas@arden.ac.uk)

[![IVY](https://img.shields.io/badge/wafa.abbas.lhr%40rootsivy.edu.pk-IVY%20College-2E8B57?style=for-the-badge&logo=microsoftoutlook&logoColor=white)](mailto:wafa.abbas.lhr@rootsivy.edu.pk)

</div>

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

*Built with ❤️ for responsible AI in mental health*

<br/>

[![🚀 Try the Live App](https://img.shields.io/badge/🚀%20Try%20the%20Live%20App%20Now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://psyche-risk-zjb4qjvfx6dg2ojv6pyft2.streamlit.app/)

</div>
