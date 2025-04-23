from preprocess import preprocess_data
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess
df_raw = pd.read_csv("sample.csv")
df_cleaned = preprocess_data(df_raw)

# Sample subset
df_sample = df_cleaned.sample(n=100000, random_state=42)

print(df_sample.head())

# Target and features
y = df_sample["total_amount"]
X = df_sample.drop(columns=["total_amount"])

# Define categorical and numerical features
categorical_cols = ["PU_DO_pair", "hour_bin", "is_rainy"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ColumnTransformer with OrdinalEncoder (since XGBoost handles ints)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
    ],
    remainder="passthrough"  # Keep all numerical columns
)

# Define XGBoost model (no GridSearch)
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Test R² : {r2_score(y_test, y_pred):.4f}")

# Save the full pipeline
joblib.dump(pipeline, "xgb_taxi_model.pkl")
