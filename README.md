🚲 Machine Learning — Bike Sharing Demand Prediction

This project explores the Bike Sharing Dataset from the Capital Bikeshare system in Washington D.C. (2011–2012).
The goal was to use supervised machine learning models to predict bike rental demand, both hourly and daily, and to evaluate whether ML models could outperform a simple baseline (average rentals).

📂 Dataset

Source: UCI Machine Learning Repository

Records:

Hourly data: 17,379 rows

Daily data: 731 rows

Features:

Weather conditions (temperature, humidity, windspeed, weather category)

Time information (season, year, month, day of week, hour)

Special days (holidays, working days)

Target variable: cnt → total number of bikes rented (casual + registered).

⚙️ Workflow

Exploratory Data Analysis (EDA): trends, seasonality, and correlations.

Preprocessing: handling normalized features, scaling, one-hot encoding.

Baseline model: mean predictor for comparison.

Models trained:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Model tuning: GridSearchCV with 5-fold cross-validation.

Evaluation metrics: MAE, RMSE, R².

Deployment: interactive demo built with Streamlit.

📊 Results
Daily (day.csv)
Model	MAE ↓	RMSE ↓	R² ↑
Linear Regression	~582	~799	0.84
Random Forest	~469	~708	0.875
Gradient Boosting	438	642	0.897
Hourly (hour.csv)
Model	MAE ↓	RMSE ↓	R² ↑
Decision Tree	~37	~62	0.877
Random Forest	~38	~54	0.906
Gradient Boosting	31	47	0.931

✅ Both datasets show 75–80% improvement over the baseline, with Gradient Boosting Regressor performing best.

🌐 Live Demo

👉 Streamlit App

(link will work after deployment)

🧠 Challenges & Learnings

Applied a complete ML workflow (EDA → preprocessing → training → evaluation).

Learned to handle normalized features.

Understood the importance of baseline comparison.

Practiced GridSearchCV with K-fold cross-validation.

🚀 Future Improvements

Add external data (e.g., events/holidays, weather forecasts).

Try deep learning models for time-series predictions.

Deploy a real-time dashboard for bike-sharing companies.

👩🏻‍💻 Built by

Sheila Géa
Multidisciplinary Designer & Data Analyst

🌐 Website
 · 💼 LinkedIn
 · 📂 GitHub

✨ This project was developed as part of a Machine Learning bootcamp project (2025).


