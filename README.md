# 🏎️ F1 Race Predictor

This project predicts Formula 1 race results using historical race data and machine learning.  
It leverages the [FastF1](https://theoehrly.github.io/Fast-F1/) library to fetch official race data and trains a classifier to predict whether a driver will finish on the **Podium**, in the **Points**, or with **No Points**.

---

## 📌 Project Overview

- Collects **historical race data** for a specific driver across multiple seasons.
- Uses the **last 3 races before each event** to build features:
  - Track name
  - Grid position
  - Finishing position
- Labels the target as:
  - 🥇 Podium (Top 3)  
  - 🔟 Points (4th–10th)  
  - ❌ No Points (11th–20th)  
- Trains a **HistGradientBoostingClassifier** with sample weighting (grid position based).

---

## ⚙️ How It Works

1. **Data Collection**  
   - `F1Predictor.generate_training_data(driver_code, year)` pulls race results from FastF1.
   - Builds a dataset of race histories with the last 3 events as features.

2. **Data Processing**  
   - One-hot encodes categorical features (track, finishing class).  
   - Converts numeric features (grid positions).  

3. **Model Training**  
   - Splits into training and validation sets.  
   - Trains a `HistGradientBoostingClassifier`.  
   - Evaluates accuracy using validation data.

