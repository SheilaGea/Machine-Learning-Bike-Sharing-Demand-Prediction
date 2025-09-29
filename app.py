import streamlit as st
import pandas as pd
import altair as alt
from joblib import load
import os

st.set_page_config(page_title="Machine Learning — Bike Sharing Demand Prediction", page_icon="🚲", layout="wide")
st.title("Machine Learning —  Bike Sharing Demand Prediction. 🚲 ")
st.markdown(" ")

# -------- Model loaders (no saving here) --------
@st.cache_resource
def load_hourly_model():
    return load("gbr_hourly_pipeline.joblib")  # required file

@st.cache_resource
def load_daily_model():
    path = "gbr_daily_pipeline.joblib"
    if os.path.exists(path):
        return load(path)  # optional file
    return None  # handle missing file gracefully

hourly_model = load_hourly_model()
daily_model  = load_daily_model()

@st.cache_data
def load_hour_csv():
    return pd.read_csv("hour.csv")

@st.cache_data
def load_day_csv():
    df = pd.read_csv("day.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df


# -------- tabs --------

tab1, tab4, tab2, tab3, tab6, tab5 = st.tabs([
    "📊 Dataset Overview",
    "📈 Results & Insights",
    "⏰ Model Demo Hourly",
    "📅 Model Demo Daily",
    "✅ Conclusion",
    "ℹ️ About"
])

# ---------------------------
# TAB 1 -  Dataset Description
# ---------------------------
with tab1:
    st.header("📊 Dataset Overview")
    st.markdown(" ") 

    # --- Section 1: Text + Image ---
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown(
            """
**The Bike Sharing Dataset** comes from the Capital Bikeshare system in Washington D.C. (2011–2012). 
 
It includes **hourly and daily records** of bike rentals, along with:

- **Weather conditions** (temperature, humidity, windspeed, weather category)  

- **Time information** (season, year, month, day of week, hour)  

- **Special days** (holidays, working days)  


**Target variable:** `cnt` → total number of bikes rented.
            """
        )

    with right:
        st.image(
            "bikeshare.jpg",
        )
        st.caption("📸 Photo: Tomi Vadász on Unsplash")

    st.markdown("---") 

    # --- Section 2: Dataset sample ---
    df = load_hour_csv()
    st.subheader("Dataset Sample")
    st.markdown(" ") 
    st.dataframe(df.head())

    st.markdown("---") 

    # --- Section 3: Features + Image ---
    st.subheader("Example Features")
    st.markdown(" ") 

    col3, col4 = st.columns([2, 1], gap="large")

    with col3:
        st.write("""
        - `season`: 1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter  
                      
        - `yr`: 0=2011, 1=2012  
                       
        - `hr`: hour of the day (0–23)  
                      
        - `weathersit`: 1 = Clear, 2 = Mist/Cloudy, 3 = Light Snow/Rain, 4 = Severe Weather  
                         
        - `temp`: Normalized temperature (°C / 41)  
                 
        - `cnt`: Total rentals (target variable)  
                 
                 
        """)

    with col4:
        st.image("bikeshare2.jpg")
        st.caption("📸 Photo: The Transport Enthusiast DC")

    
    st.markdown("---") 

    # --- Section 4: Daily trend (from day.csv) ---

    st.subheader("Daily Rentals Over Time (2011–2012)")

    st.markdown("")  

    # Load daily dataset
    df_day = load_day_csv()
    df_day = df_day.sort_values("dteday").reset_index(drop=True)
    df_day["cnt_ma7"] = df_day["cnt"].rolling(window=7, center=True).mean()

    # Melt data for Altair
    df_melted = df_day.melt(
        id_vars=["dteday"], 
        value_vars=["cnt", "cnt_ma7"],
        var_name="Type", 
        value_name="Rentals"
    )

    # Define custom colors
    color_scale = alt.Scale(
        domain=["cnt", "cnt_ma7"],
        range=["#1f77b4", "#ff7f0e"]  # Blue, Orange
    )

    chart = alt.Chart(df_melted).mark_line().encode(
        x="dteday:T",
        y="Rentals:Q",
        color=alt.Color(
            "Type:N", 
            scale=color_scale,
            legend=alt.Legend(
                title="Series",
                labelExpr="datum.label == 'cnt' ? 'Daily Rentals' : '7-Day Avg'"
            )
        )
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")   

    # --- Section 5: Monthly average rentals (Altair bar chart) ---

    st.subheader("Monthly Average Rentals")
    st.markdown(" ") 

    # Compute monthly averages from df_day (already loaded above)
    monthly = (
        df_day.assign(year_month=df_day["dteday"].dt.to_period("M"))
            .groupby("year_month")["cnt"].mean()
            .to_timestamp()
            .reset_index()
            .rename(columns={"year_month": "month", "cnt": "avg_cnt"})
    )

    # Altair bar chart (orange bars to match the theme)
    bar = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("avg_cnt:Q", title="Average Rentals"),
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip("avg_cnt:Q", title="Avg Rentals", format=".0f"),
            ],
            color=alt.value("#ff7f0e")  # Orange
        )
        .properties(width=700, height=300)
    )

    st.altair_chart(bar, use_container_width=True)

    st.markdown(
    """
    **Insight:**  
    - Bike rentals show **strong seasonality**.  
    - **Summer months** (June–August) have the highest demand.  
    - **Winter months** (December–February) have much lower demand.  
    - This seasonal pattern is important for the ML model — it learns these trends and helps predict future demand.
    """
)

    st.markdown("---") 
    st.markdown("Created by Sheila Géa")

# ---------------------------
# TAB 2 - Model Demo Hourly
# ---------------------------

with tab2:
    st.header("⏰ Model Demo — Hourly Prediction")
    st.markdown(" ") 
    st.write("Enter weather and calendar info to predict **rentals for a specific time**.")
   
    # --- Layout: two columns for inputs ---
    c1, c2 = st.columns(2)

    with c1:
        yr = st.selectbox("Year", [0, 1], index=1, format_func=lambda x: "2011" if x == 0 else "2012")
        season = st.selectbox("Season", [1, 2, 3, 4],
                              format_func=lambda x: {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}[x])
        mnth = st.slider("Month (1–12)", 1, 12, 7)
        hr = st.slider("Hour (0–23)", 0, 23, 17)
        weekday = st.slider("Weekday (0=Sun - 6=Sat)", 0, 6, 3)

    with c2:
        holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        workingday = st.selectbox("Working day?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        weathersit = st.selectbox("Weather (1–4)", [1, 2, 3, 4],
                                  format_func=lambda x: {
                                      1:"Clear/Few clouds",
                                      2:"Mist/Cloudy",
                                      3:"Light Snow/Rain",
                                      4:"Heavy Rain/Snow"
                                  }[x])

        st.markdown("**Weather (real-world units)**")
        temp_c = st.slider("Temperature °C (0–41)", 0.0, 41.0, 20.0, 0.5)
        atemp_c = st.slider("Feels-like °C (0–50)", 0.0, 50.0, 22.0, 0.5)
        hum_pct = st.slider("Humidity % (0–100)", 0.0, 100.0, 60.0, 1.0)
        wind_raw = st.slider("Wind speed scale (0–67)", 0.0, 67.0, 10.0, 0.5)

    # Convert to dataset's normalized scale expected by the model
    temp = temp_c / 41.0
    atemp = atemp_c / 50.0
    hum = hum_pct / 100.0
    windspeed = wind_raw / 67.0

    # Build the single-row DataFrame with EXACT training feature names
    input_row = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }])

    

    st.markdown("---") 

    st.markdown("#### Model Input Preview")
    st.markdown(" ") 
    st.dataframe(input_row, use_container_width=True)

    if st.button("🔮 Predict hourly rentals"):
        pred = hourly_model.predict(input_row)[0]  
        st.success(f"Estimated rentals this hour: **{int(round(pred))}** bikes")
        st.write("* Prediction from the tuned Gradient Boosting pipeline trained on hourly data (2011–2012).")

    st.markdown("---") 
    st.markdown("Created by Sheila Géa")


# ---------------------------
# TAB 3  - Model Demo Daily
# ---------------------------

with tab3:
    st.header("📅 Model Demo — Daily Prediction")
    st.markdown(" ")
    st.write("Enter weather and calendar info to predict **total rentals for a day**.")

    if daily_model is None:
        st.warning(
            "Daily model file `gbr_daily_pipeline.joblib` not found. "
            "Please save it from your notebook with:\n\n"
            "`dump(best_gbr_daily, 'gbr_daily_pipeline.joblib')`"
        )
    else:
        # Two columns for inputs (no 'hr' in daily features)
        d1, d2 = st.columns(2)

        with d1:
            yr_d = st.selectbox(
                "Year", [0, 1], index=1,
                format_func=lambda x: "2011" if x == 0 else "2012",
                key="year_daily"
            )
            season_d = st.selectbox(
                "Season", [1, 2, 3, 4],
                format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}[x],
                key="season_daily"
            )
            mnth_d = st.slider("Month (1–12)", 1, 12, 7, key="month_daily")
            weekday_d = st.slider("Weekday (0=Sun - 6=Sat)", 0, 6, 3, key="weekday_daily")
            holiday_d = st.selectbox(
                "Holiday?", [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                key="holiday_daily"
            )

        with d2:
            workingday_d = st.selectbox(
                "Working day?", [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                key="workingday_daily"
            )
            weathersit_d = st.selectbox(
                "Weather (1–4)", [1, 2, 3, 4],
                format_func=lambda x: {
                    1:"Clear/Few clouds",
                    2:"Mist/Cloudy",
                    3:"Light Snow/Rain",
                    4:"Heavy Rain/Snow"
                }[x],
                key="weathersit_daily"
            )

            st.markdown("**Weather (real-world units)**")
            temp_c_d = st.slider("Avg Temperature °C (0–41)", 0.0, 41.0, 20.0, 0.5, key="temp_daily")
            atemp_c_d = st.slider("Avg Feels-like °C (0–50)", 0.0, 50.0, 22.0, 0.5, key="atemp_daily")
            hum_pct_d = st.slider("Avg Humidity % (0–100)", 0.0, 100.0, 60.0, 1.0, key="hum_daily")
            wind_raw_d = st.slider("Avg Wind speed scale (0–67)", 0.0, 67.0, 10.0, 0.5, key="wind_daily")

        # Normalize to dataset scale (same as training)
        temp_d = temp_c_d / 41.0
        atemp_d = atemp_c_d / 50.0
        hum_d = hum_pct_d / 100.0
        windspeed_d = wind_raw_d / 67.0

        # Build daily input row (NO 'hr' for daily)
        daily_input = pd.DataFrame([{
            "season": season_d,
            "yr": yr_d,
            "mnth": mnth_d,
            "holiday": holiday_d,
            "weekday": weekday_d,
            "workingday": workingday_d,
            "weathersit": weathersit_d,
            "temp": temp_d,
            "atemp": atemp_d,
            "hum": hum_d,
            "windspeed": windspeed_d
        }])

        st.markdown("---")
        st.markdown("#### Model Input Preview (Daily)")
        st.dataframe(daily_input, use_container_width=True)

        if st.button("🔮 Predict daily rentals", key="predict_daily"):
            pred_day = daily_model.predict(daily_input)[0]
            st.success(f"Estimated rentals for the day: **{int(round(pred_day))}** bikes")
            st.write("* Prediction from the tuned Gradient Boosting pipeline trained on daily data (2011–2012).")

    st.markdown("---") 
    st.markdown("Created by Sheila Géa")


# ---------------------------
# TAB 4 - Results & Insights
# ---------------------------

with tab4:
    st.header("📈 Results & Insights")
    st.markdown(" ")

    # === New Intro ===
    st.markdown(
        """
        To evaluate bike rental demand, I trained and tested multiple **supervised regression models**
        on both the **daily** and **hourly** datasets.  
        
        The workflow included:
        - Train/test split and preprocessing (scaling + one-hot encoding).  
        - A simple **baseline** (predicting the mean) for comparison.  
        - Training of several regression models:  
          - **Linear Regression**  
          - **Decision Tree Regressor**  
          - **Random Forest Regressor**  
          - **Gradient Boosting Regressor**  
        - Hyperparameter tuning with **GridSearchCV + 5-fold cross-validation** for the top 3 models.  

        The goal was to minimize prediction error (MAE/RMSE) and maximize explained variance (R²).
        """
    )

    st.markdown("---")

    # ===== 1) Quick Summary cards =====
    st.subheader("📌 Summary vs Baseline")
    st.markdown(" ")

    # Reported metrics (from your notebook runs)
    # Baselines
    baseline_mae_daily = 1711.99   # ~1712
    baseline_mae_hourly = 140.08   # ~140

    # Best models (tuned)
    daily_best_mae  = 438.49
    daily_best_rmse = 641.99
    daily_best_r2   = 0.8972

    hourly_best_mae  = 30.62
    hourly_best_rmse = 46.69
    hourly_best_r2   = 0.9311

    # Improvements
    daily_impr  = (baseline_mae_daily - daily_best_mae) / baseline_mae_daily * 100
    hourly_impr = (baseline_mae_hourly - hourly_best_mae) / baseline_mae_hourly * 100

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily (day.csv)**")
        st.metric(
            label="MAE (Baseline → Best GBR)",
            value=f"{daily_best_mae:.0f}",
            delta=f"-{daily_impr:.1f}% vs baseline (~{baseline_mae_daily:.0f})"
        )
        st.caption(f"Best GBR: RMSE ~ {daily_best_rmse:.0f}, R² ~ {daily_best_r2:.2f}")

    with c2:
        st.markdown("**Hourly (hour.csv)**")
        st.metric(
            label="MAE (Baseline → Best GBR)",
            value=f"{hourly_best_mae:.1f}",
            delta=f"-{hourly_impr:.1f}% vs baseline (~{baseline_mae_hourly:.0f})"
        )
        st.caption(f"Best GBR: RMSE ~ {hourly_best_rmse:.0f}, R² ~ {hourly_best_r2:.2f}")

    st.markdown("---")

        
    # ===== 2) Detailed tables =====
    st.subheader("🔬 Tuned Models — Test Set Performance")
    st.markdown(" ")

    # Daily top 3 (tuned)
    daily_table = pd.DataFrame([
        {"Model": "Linear Regression ", "MAE ↓": 582.47, "RMSE ↓": 798.89, "R² ↑": 0.8408},
        {"Model": "Random Forest ",     "MAE ↓": 469.02, "RMSE ↓": 708.01, "R² ↑": 0.8750},
        {"Model": "Gradient Boosting ", "MAE ↓": daily_best_mae, "RMSE ↓": daily_best_rmse, "R² ↑": daily_best_r2},
    ])

    # Hourly top 3 (tuned)
    hourly_table = pd.DataFrame([
        {"Model": "Decision Tree ",     "MAE ↓": 37.25, "RMSE ↓": 62.37, "R² ↑": 0.8772},
        {"Model": "Random Forest ",     "MAE ↓": 38.43, "RMSE ↓": 54.45, "R² ↑": 0.9064},
        {"Model": "Gradient Boosting ", "MAE ↓": hourly_best_mae, "RMSE ↓": hourly_best_rmse, "R² ↑": hourly_best_r2},
    ])

    # Format numbers: integers for MAE/RMSE, 3 decimals for R²
    def format_tables(df):
        df = df.copy()
        df["MAE ↓"]  = df["MAE ↓"].round(0).astype(int)
        df["RMSE ↓"] = df["RMSE ↓"].round(0).astype(int)
        df["R² ↑"]   = df["R² ↑"].round(3)   # force 3 decimals only
        return df

    daily_table_fmt  = format_tables(daily_table)
    hourly_table_fmt = format_tables(hourly_table)

    # Highlight GBR row in orange + bold
    def highlight_gbr(row):
        if "Gradient Boosting" in row["Model"]:
            return ['color: #ff7f0e; font-weight: 700'] * len(row)
        return [''] * len(row)

    # Bigger, cleaner table styles
    base_styles = [
        {'selector': 'th', 'props': 'font-size:16px; font-weight:700; padding:8px 12px;'},
        {'selector': 'td', 'props': 'font-size:16px; padding:8px 12px;'},
        {'selector': 'tbody tr:nth-child(even)', 'props': 'background-color: rgba(255,255,255,0.03);'}
    ]

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Daily**")
        styled_daily = (daily_table_fmt
                        .style
                        .hide(axis="index")
                        .apply(highlight_gbr, axis=1)
                        .set_table_styles(base_styles)
                        .format(precision=3))  # enforce decimals
        st.table(styled_daily)

    with c4:
        st.markdown("**Hourly**")
        styled_hourly = (hourly_table_fmt
                        .style
                        .hide(axis="index")
                        .apply(highlight_gbr, axis=1)
                        .set_table_styles(base_styles)
                        .format(precision=3))  # enforce decimals
        st.table(styled_hourly)

    st.markdown("---")

    # ===== 3) Visual comparison (Altair) =====

    # === Daily — MAE (horizontal bar) ===
    st.subheader("📊 Visual Comparison — Daily (MAE)")
    st.markdown(" ")

    daily_mae_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
        "MAE":   [582.47, 469.02, 438.49]  # your daily tuned results
    })

    daily_mae_chart = (
        alt.Chart(daily_mae_df, title="Daily")
        .mark_bar(size=32)
        .encode(
            y=alt.Y("Model:N", sort="-x", title=None),
            x=alt.X("MAE:Q", title="MAE (lower is better)"),
            color=alt.value("#1f77b4"),  # blue
            tooltip=["Model", alt.Tooltip("MAE:Q", format=".1f")]
        )
        .properties(height=240)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_title(fontSize=16)
    )

    st.altair_chart(daily_mae_chart, use_container_width=True)

    st.markdown("---")

    # === Hourly — MAE (horizontal bar) ===
    st.subheader("📊 Visual Comparison — Hourly (MAE)")
    st.markdown(" ")

    hourly_mae_df = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "Gradient Boosting"],
        "MAE":   [37.25, 38.43, hourly_best_mae]
    })

    mae_chart = (
        alt.Chart(hourly_mae_df, title="Hourly")
        .mark_bar(size=32)
        .encode(
            y=alt.Y("Model:N", sort="-x", title=None),
            x=alt.X("MAE:Q", title="MAE (lower is better)"),
            color=alt.value("#ff7f0e"),
            tooltip=["Model", alt.Tooltip("MAE:Q", format=".1f")]
        )
        .properties(height=240)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_title(fontSize=16)
    )
    st.altair_chart(mae_chart, use_container_width=True)

    st.markdown("---")

    
    # ===== 4) Interpretation / Talking points =====

    st.subheader("🧠 Interpretation")
    st.markdown("""
    - **Both datasets beat the baseline by ~75–80%** in MAE — strong improvement.
    - **Gradient Boosting** is the best overall on both daily and hourly data.
    - **Hourly model** achieves very low error (MAE ≈ 31 bikes/hour, R² ≈ 0.93), great for **operational decisions** (rebalancing during rush hours).
    - **Daily model** is also strong (MAE ≈ 438 bikes/day, R² ≈ 0.90), good for **planning & staffing**.
    - Small differences after tuning (e.g., RF R² drop) are **normal** due to CV–test split variance.
    """)

    st.caption("Note: All results are from Washington D.C. Bike Sharing (2011–2012) datasets.")


    st.markdown("---") 
    st.markdown("Created by Sheila Géa")


# ---------------------------
# TAB 5 - About
# ---------------------------
with tab5:
    st.header("ℹ️ About This Project")
    st.markdown(" ") 


    # --- Two columns layout ---
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("""
This project explores the **Bike Sharing Dataset** from the Capital Bikeshare system in Washington D.C. (2011–2012).  
The goal was to predict bike rental demand using **supervised machine learning models**.

**Source:**  
                    
Fanaee-T, Hadi, and Gama, João, “Event labeling combining ensemble detectors and background knowledge”,  
*Progress in Artificial Intelligence* (2013).  

**UCI ML Repository:**  
                    
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
""")

    with right:
        st.image("bikeshare3.jpg")
        st.caption("📸 Photo: Pascal Müller on Unsplash")

    st.markdown("---")

    st.subheader("🧠 Challenges & Learnings")
    st.markdown("""
- Applied a complete ML workflow: **EDA → preprocessing → training → evaluation**; 
- Understood and handled **normalized features** (temp, atemp, humidity, windspeed);  
- Compared models against a **baseline** (mean predictor) to prove value;
- Practiced **GridSearchCV** with **K-fold cross-validation**.
""")

    st.markdown("---")

    st.subheader("🚀 Ideas for Future Improvements")
    st.markdown("""

- Add external signals (holidays/events calendar, **weather forecast**);
- Deploy a live **dashboard** for real-time demand prediction; 
- Try deep learning models (e.g., Neural Networks) for time-series data.  
     
""")

    st.markdown("---")

    st.header("👩🏻‍💻 Built by")
    st.subheader("Sheila Géa")
    st.markdown("""
Multidisciplinary Designer & Data Analyst  

🌐 **Website:** [sheiladgea.com](https://sheiladgea.com)  
💼 **LinkedIn:** [linkedin.com/in/sheilagea](https://www.linkedin.com/in/sheilagea)  
📂 **GitHub:** [github.com/sheilagea](https://github.com/sheilagea)  
""")


    st.markdown("---") 
    
# ---------------------------
# TAB 6 - Conclusion
# ---------------------------
with tab6:
    st.header("✅ Conclusion:")
    st.markdown(" ") 

    st.markdown("""
### 🎯 Project Goal
The goal of this project was to **predict bike rental demand** using supervised machine learning,  
and evaluate whether ML models could outperform a simple baseline (average rentals).

---

### 🚀 Key Result
The machine learning models — especially **Gradient Boosting** — achieved strong predictive accuracy,  
reducing forecast errors by over **75% compared to baseline predictions**.

---

### 📌 How this creates value for bike sharing companies in Washington D.C.:

- 🚲 **Operational efficiency:** Better allocation of bikes across stations based on predicted demand.  
- 😀 **Customer satisfaction:** Reduces empty stations and overcrowded ones, improving user experience.  
- 🛠️ **Strategic planning:** Seasonal and hourly insights support staffing, pricing, and maintenance decisions.  
- 🌍 **Sustainability:** Reliable service encourages bike usage, helping reduce car traffic and emissions.  

---

### ✅ Takeaway
By moving from a simple baseline predictor to advanced ML models,  
bike sharing companies can **optimize resources, improve service quality,  
and contribute to a greener urban mobility system**.
""")

    st.markdown("---") 
    st.markdown("Created by Sheila Géa")