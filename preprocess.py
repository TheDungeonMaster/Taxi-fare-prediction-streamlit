# preprocess.py
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Point

def preprocess_data(df):
    df_cleaned = df.drop(columns=[
        "VendorID", "RatecodeID", "tpep_dropoff_datetime", "fare_amount", "tip_amount", "tolls_amount",
        "improvement_surcharge", "store_and_fwd_flag", "extra", "mta_tax",
    ])

    zones = gpd.read_file("taxi_zones/taxi_zones.shp")
    zones = zones[["LocationID", "geometry"]].to_crs("EPSG:4326")

    def map_location_to_zone(df, lon_col, lat_col, zone_df, zone_name):
        coords = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        geo_df = gpd.GeoDataFrame(df[[lon_col, lat_col]].copy(), geometry=coords, crs="EPSG:4326")
        joined = gpd.sjoin(geo_df, zone_df, how="left", predicate="within")
        return joined["LocationID"].rename(zone_name)

    df_cleaned["PULocationID"] = map_location_to_zone(df_cleaned, "pickup_longitude", "pickup_latitude", zones, "PULocationID")
    df_cleaned["DOLocationID"] = map_location_to_zone(df_cleaned, "dropoff_longitude", "dropoff_latitude", zones, "DOLocationID")

    df_cleaned["pickup_hour"] = pd.to_datetime(df_cleaned["tpep_pickup_datetime"]).dt.hour
    df_cleaned["pickup_dayofweek"] = pd.to_datetime(df_cleaned["tpep_pickup_datetime"]).dt.dayofweek
    df_cleaned["is_weekend"] = df_cleaned["pickup_dayofweek"].isin([5, 6]).astype(int)
    df_cleaned["is_rush_hour"] = df_cleaned["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    def haversine_vectorized(lat1, lon1, lat2, lon2):
        R = 6371
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df_cleaned["haversine_distance"] = haversine_vectorized(
        df_cleaned["pickup_latitude"], df_cleaned["pickup_longitude"],
        df_cleaned["dropoff_latitude"], df_cleaned["dropoff_longitude"]
    ).round(2)

    AIRPORT_ZONE_IDS = {"JFK": 132, "LaGuardia": 138, "Newark": 1}
    airport_ids = list(AIRPORT_ZONE_IDS.values())
    df_cleaned["is_airport"] = (
        df_cleaned["PULocationID"].isin(airport_ids) | df_cleaned["DOLocationID"].isin(airport_ids)
    ).astype(int)

    LAT, LON = 40.7128, -74.0060
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": "2016-03-01", "end_date": "2016-03-31",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    }
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_data = requests.get(weather_url, params=params).json()

    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather_data["hourly"]["time"]),
        "temperature": weather_data["hourly"]["temperature_2m"],
        "humidity": weather_data["hourly"]["relative_humidity_2m"],
        "precipitation": weather_data["hourly"]["precipitation"],
        "wind_speed": weather_data["hourly"]["wind_speed_10m"]
    })
    weather_df["datetime"] = weather_df["datetime"].dt.floor("h")
    df_cleaned["pickup_hour"] = pd.to_datetime(df_cleaned["tpep_pickup_datetime"]).dt.floor("h")
    df_cleaned = df_cleaned.merge(weather_df, left_on="pickup_hour", right_on="datetime", how="left")

    df_cleaned = df_cleaned.drop(columns=[
        "tpep_pickup_datetime", "datetime", "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude", 'passenger_count', 'payment_type', 'Unnamed: 0'
    ])

    df_cleaned["PU_DO_pair"] = df_cleaned["PULocationID"].astype(str) + "_" + df_cleaned["DOLocationID"].astype(str)
    df_cleaned["distance_ratio"] = (df_cleaned["trip_distance"] / df_cleaned["haversine_distance"].replace(0, np.nan)).round(2)
    df_cleaned["is_rainy"] = (df_cleaned["precipitation"] > 0.5).astype(int)

    def hour_to_bin(hour):
        if 5 <= hour <= 10:
            return "morning"
        elif 11 <= hour <= 15:
            return "midday"
        elif 16 <= hour <= 20:
            return "evening"
        else:
            return "night"

    df_cleaned["hour_bin"] = df_cleaned["pickup_hour"].dt.hour.apply(hour_to_bin)
    df_cleaned["pickup_hour"] = df_cleaned["pickup_hour"].dt.hour
    df_cleaned = df_cleaned.drop(columns=["PULocationID", "DOLocationID"])

    return df_cleaned.dropna()
