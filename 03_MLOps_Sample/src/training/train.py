import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load data
df = pd.read_csv("data/data.csv")
X = df[["price_lag1", "price_lag2"]]
y = df["price_today"]

# 2. Split train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 3. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# 5. Simpan model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
