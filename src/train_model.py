import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../data/train.csv")

# Preprocessing
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Save model + scaler
joblib.dump(model, "../models/house_price_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

# Evaluate
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))

print("Training complete! Model saved in models/")
