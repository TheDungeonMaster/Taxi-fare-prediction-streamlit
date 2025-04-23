# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

tab1, tab2, tab3, tab4 = st.tabs(["Estimate Fare", " Fare Map by Zone", " Best Time to Ride", "What if simulator"])


# Load model pipeline
@st.cache_resource
def load_model():
    return joblib.load("xgb_taxi_model.pkl")

model = load_model()

with tab1:
    st.title("NYC Taxi Fare Estimator")

    st.markdown("""
        <style>
            .fare-box {
                position: fixed;
                bottom: 30px;
                right: 30px;
                background-color: #fffae6;
                padding: 16px 24px;
                border-radius: 12px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
                font-size: 20px;
                z-index: 9999;
                border: 1px solid #ffcc00;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)



    st.subheader("Enter Ride Details")

    # === Pickup & Dropoff Zone (manual entry) ===
    pickup_zone = "1"
    dropoff_zone = "1"

    # Combine them
    pu_do_pair = f"{pickup_zone.strip()}_{dropoff_zone.strip()}"

    # Continue as before...
    hour_bin = "midday"
    pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, value=8)
    pickup_dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, value=2.5)
    haversine_distance = trip_distance
    temperature = st.slider("Temperature (°C)", min_value=-10, max_value=40, value=15)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
    precipitation = st.slider("Precipitation (mm)", min_value=0.0, max_value=20.0, value=0.0)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)

    is_weekend = True if pickup_dayofweek == 5 or pickup_dayofweek ==6 else False
    is_rush_hour = st.checkbox("Rush Hour?", value=True)
    is_airport = st.checkbox("From/To Airport?", value=False)
    is_rainy = precipitation > 0.5


    distance_ratio = round(trip_distance / (haversine_distance if haversine_distance else 1), 2)

    # === Construct Feature DataFrame ===
    input_df = pd.DataFrame([{
        "PU_DO_pair": pu_do_pair,
        "hour_bin": hour_bin,
        "pickup_hour": pickup_hour,
        "pickup_dayofweek": pickup_dayofweek,
        "trip_distance": trip_distance,
        "haversine_distance": haversine_distance,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "is_weekend": int(is_weekend),
        "is_rush_hour": int(is_rush_hour),
        "is_airport": int(is_airport),
        "distance_ratio": distance_ratio,
        "is_rainy": int(is_rainy)
    }])

    # === Predict Fare Immediately on Input Change ===
    prediction = model.predict(input_df)[0]
    st.markdown(f"""
        <div class="fare-box">
            💰 <b>Estimated Fare:</b> ${prediction:.2f}
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Predicted Fare by Hour of Day")

    st.markdown("Visualize how predicted fares change throughout the day for a given route and weather conditions.")

    # --- User inputs ---
    pickup_zone = st.number_input("Pickup Zone ID", min_value=1, max_value=263, value=132, key="pickup_zone_trend")
    dropoff_zone = st.number_input("Dropoff Zone ID", min_value=1, max_value=263, value=138, key="dropoff_zone_trend")
    hour_bin = st.selectbox("Time Bin", ["morning", "midday", "evening", "night"], key="hour_bin_trend")
    trip_distance = st.slider("Trip Distance (mi)", 0.5, 20.0, 2.5, key="trip_distance_trend")
    haversine_distance = st.slider("Haversine Distance (km)", 0.5, 20.0, 2.2, key="haversine_trend")
    is_rainy = st.checkbox("Raining?", value=False, key="rain_trend")
    is_rush_hour = st.checkbox("Rush Hour?", value=False, key="rush_trend")
    is_weekend = st.checkbox("Weekend?", value=False, key="weekend_trend")
    is_airport = st.checkbox("From/To Airport?", value=False, key="airport_trend")

    # --- Generate data for 24 hours ---
    hours = list(range(24))
    df_trend = pd.DataFrame({
        "PU_DO_pair": [f"{pickup_zone}_{dropoff_zone}"] * 24,
        "hour_bin": [hour_bin] * 24,
        "pickup_hour": hours,
        "pickup_dayofweek": [2] * 24,
        "trip_distance": [trip_distance] * 24,
        "haversine_distance": [haversine_distance] * 24,
        "temperature": [15] * 24,
        "humidity": [60] * 24,
        "precipitation": [1.0 if is_rainy else 0.0] * 24,
        "wind_speed": [10] * 24,
        "is_weekend": [int(is_weekend)] * 24,
        "is_rush_hour": [int(is_rush_hour)] * 24,
        "is_airport": [int(is_airport)] * 24,
        "distance_ratio": [round(trip_distance / (haversine_distance if haversine_distance else 1), 2)] * 24,
        "is_rainy": [int(is_rainy)] * 24
    })

    # --- Predict fare for each hour ---
    df_trend["fare_pred"] = model.predict(df_trend)

    # --- Plot the trend line ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_trend["pickup_hour"], df_trend["fare_pred"], marker="o", color="orange")
    ax.set_title(f"Fare vs. Hour (Pickup: {pickup_zone} → Dropoff: {dropoff_zone})", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Fare ($)")
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    st.header(" Best Time to Ride")

    st.markdown("Pick a route and see which hour of the day offers the lowest predicted fare under fixed conditions.")

    pickup_zone = st.number_input("Pickup Zone ID", min_value=1, max_value=263, value=132, key="pickup_zone_besttime")
    dropoff_zone = st.number_input("Dropoff Zone ID", min_value=1, max_value=263, value=138, key="dropoff_zone_besttime")

    is_rainy = st.checkbox("Raining?", value=False, key="rain_besttime")
    is_rush_hour = st.checkbox("Assume Rush Hour?", value=False, key="rush_besttime")
    is_weekend = st.checkbox("Weekend?", value=False, key="weekend_besttime")
    hour_bin = st.selectbox("Fixed Time Bin (optional)", ["morning", "midday", "evening", "night"], key="hour_bin_besttime")

    # Distance constants
    trip_distance = 2.5
    haversine_distance = 2.2
    distance_ratio = trip_distance / haversine_distance

    hours = list(range(24))
    df_hours = pd.DataFrame({
        "PU_DO_pair": [f"{pickup_zone}_{dropoff_zone}"] * 24,
        "hour_bin": [hour_bin] * 24,
        "pickup_hour": hours,
        "pickup_dayofweek": [5 if is_weekend else 2] * 24,
        "trip_distance": [trip_distance] * 24,
        "haversine_distance": [haversine_distance] * 24,
        "temperature": [15] * 24,
        "humidity": [60] * 24,
        "precipitation": [1.0 if is_rainy else 0.0] * 24,
        "wind_speed": [10] * 24,
        "is_weekend": [int(is_weekend)] * 24,
        "is_rush_hour": [int(is_rush_hour)] * 24,
        "is_airport": [0] * 24,
        "distance_ratio": [distance_ratio] * 24,
        "is_rainy": [int(is_rainy)] * 24
    })

    df_hours["fare_pred"] = model.predict(df_hours)

    # Get cheapest & priciest hour
    cheapest_hour = df_hours.loc[df_hours["fare_pred"].idxmin(), "pickup_hour"]
    most_expensive_hour = df_hours.loc[df_hours["fare_pred"].idxmax(), "pickup_hour"]

    st.markdown(f"🟢 **Cheapest hour**: {cheapest_hour}:00")
    st.markdown(f"🔴 **Most expensive hour**: {most_expensive_hour}:00")

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_hours["pickup_hour"], df_hours["fare_pred"], marker="o", color="teal")
    ax.set_title(f"Fare Throughout Day (Pickup: {pickup_zone} → Dropoff: {dropoff_zone})", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Fare ($)")
    ax.grid(True)

    # Highlight min/max
    ax.axvline(x=cheapest_hour, color="green", linestyle="--", label="Cheapest")
    ax.axvline(x=most_expensive_hour, color="red", linestyle="--", label="Most Expensive")
    ax.legend()

    st.pyplot(fig)


with tab4:
    st.header(" What-If Fare Simulator")

    st.markdown("Explore how changing a single input affects predicted fare, keeping everything else constant.")

    # --- Fixed inputs ---
    pickup_zone = st.number_input("Pickup Zone ID", min_value=1, max_value=263, value=132, key="pickup_sim")
    dropoff_zone = st.number_input("Dropoff Zone ID", min_value=1, max_value=263, value=138, key="dropoff_sim")
    base_trip_distance = st.slider("Trip Distance (mi)", 0.5, 20.0, 2.5, key="trip_sim")
    base_hour = st.slider("Pickup Hour", 0, 23, 8, key="hour_sim")
    variable_to_vary = st.selectbox(
        "Select Variable to Vary",
        options=["pickup_hour", "trip_distance", "precipitation", "is_rush_hour", "is_rainy", "temperature"],
        key="var_sim"
    )

    # --- Create variation ---
    if variable_to_vary in ["pickup_hour"]:
        var_range = list(range(0, 24))
    elif variable_to_vary == "trip_distance":
        var_range = np.linspace(0.5, 20, 30)
    elif variable_to_vary == "precipitation":
        var_range = np.linspace(0, 20, 30)
    elif variable_to_vary == "temperature":
        var_range = np.linspace(-10, 40, 30)
    elif variable_to_vary in ["is_rainy", "is_rush_hour"]:
        var_range = [0, 1]
    else:
        var_range = [0]

    # --- Create input df with varying column ---
    df_sim = pd.DataFrame({
        "PU_DO_pair": [f"{pickup_zone}_{dropoff_zone}"] * len(var_range),
        "hour_bin": ["morning"] * len(var_range),
        "pickup_hour": [base_hour] * len(var_range),
        "pickup_dayofweek": [2] * len(var_range),
        "trip_distance": [base_trip_distance] * len(var_range),
        "haversine_distance": [base_trip_distance * 0.9] * len(var_range),
        "temperature": [15] * len(var_range),
        "humidity": [60] * len(var_range),
        "precipitation": [0.0] * len(var_range),
        "wind_speed": [10] * len(var_range),
        "is_weekend": [0] * len(var_range),
        "is_rush_hour": [0] * len(var_range),
        "is_airport": [0] * len(var_range),
        "distance_ratio": [1.1] * len(var_range),
        "is_rainy": [0] * len(var_range),
    })

    # Overwrite the variable to vary
    df_sim[variable_to_vary] = var_range

    # Predict
    df_sim["fare_pred"] = model.predict(df_sim)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(var_range, df_sim["fare_pred"], marker="o", color="purple")
    ax.set_title(f"Fare Sensitivity to '{variable_to_vary}'", fontsize=14)
    ax.set_xlabel(variable_to_vary)
    ax.set_ylabel("Predicted Fare ($)")
    ax.grid(True)
    st.pyplot(fig)
