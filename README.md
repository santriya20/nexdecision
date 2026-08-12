# ⚡ NexDecision: Multi-Category AI Decision Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nexdecision-w4sxpqy6g8y5pjr6b2xps7.streamlit.app/)

---

An intelligent Multi-Criteria Decision Analysis (MCDA) web application designed to help consumers eliminate decision fatigue. Built with **Streamlit**, **SQLite**, and **Scikit-Learn**, NexDecision combines **MAUT (Multi-Attribute Utility Theory)** with an embedded **Random Forest Regressor** and **Natural Language Intent Parsing** to evaluate laptops, smartphones, and cars.

---

## 🚀 Key Features

* **Natural Language Intent Parsing:** Users can type plain-English queries (e.g., *"Best smartphone for college under 40k with fast charging"*), and the engine automatically routes the category, extracts budget limits, and assigns attribute weights.

* **Multi-Category Architecture:** Dynamic switching across distinct relational tables (`laptops`, `smartphones`, `cars`) stored in SQLite.

* **MAUT Scoring Core:** Benefit/Cost min-max normalization that weighs performance, endurance, portability, and value dynamically.

* **Algorithmic Fair-Price Valuation:** Uses an embedded `RandomForestRegressor` to estimate fair market value and highlight products priced below predicted cost.

* **Hardware Bottleneck Detection:** Rule-based heuristics flag performance mismatches (e.g., high-end CPU throttled by 8GB RAM, fast phones paired with slow charging speeds).

* **Side-by-Side Visual Benchmarks:** Clear horizontal percentage bar charts comparing the top two ranked products across all evaluation criteria.

* **Data Export:** Export customized ranked recommendations directly to CSV.

---

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit, Plotly Graph Objects
* **Machine Learning & Math:** Scikit-Learn (Random Forest), NumPy, Pandas
* **Database:** SQLite3
* **Language & SDKs:** Python 3.10+, Google GenAI SDK (optional LLM enhancement)

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit application & decision engine
├── nexdecision_multiproduct.db # SQLite database (auto-generated on first run)
├── requirements.txt           # Production dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## ⚡ Quickstart & Local Setup

1. Clone the repository

```text
git clone https://github.com/santriya20/nexdecision.git
cd nexdecision
```

2. Set up virtual environment & install dependencies

```powershell
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. Run the Streamlit App

```powershell
streamlit run app.py
```

---

## 🧮 Mathematical & Decision Formulation

NexDecision computes individual utility scores using **Min-Max Normalization**:

* **Benefit Attribute** *(Higher is better — e.g., CPU, RAM, Battery, Mileage, Safety)*:
  $$\text{Benefit Attribute: } x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

* **Cost / Price Attribute** *(Lower is better — e.g., Price, Weight)*:
  $$\text{Cost / Price Attribute: } x' = \frac{\max(x) - x}{\max(x) - \min(x)}$$

The final ranking score is calculated using **Weighted Aggregation (MAUT)**:

$$\text{Total Score} = \frac{\sum (w_i \cdot x'_i)}{\sum w_i}$$

---

## 📄 License
MIT License. Free for educational and commercial use.

