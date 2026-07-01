import pandas as pd

# Load dataset
df = pd.read_csv("../datasets/housing.csv")

features_names = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

X = df[features_names]
y = df["median_house_value"]

# Handle missing values
X["total_bedrooms"] = X["total_bedrooms"].fillna(
    X["total_bedrooms"].mean()
)

# Standardize each feature
X_scaled = (X - X.mean()) / X.std()

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)



# --------------------------------------------


from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

true_model = LinearRegression(n_features=X_scaled.shape[1], lr=0.01)
loss_fn = MSEloss()

t_loss = 0
for epoch in range(1000):
    y_pred = true_model.forward(X_train)

    t_loss = loss_fn(y_train, y_pred)
    d_loss = loss_fn.grad(y_train, y_pred)

    dw, db = true_model.grad(X_train, d_loss)

    true_model.backward(dw, db)

trueml_y_pred = true_model.weights @ X_test.T + true_model.bias
true_mse = loss_fn(y_test, trueml_y_pred)

print(f"scikit-learn MSE : {mse:.4f}")
print(f"TrueML MSE       : {true_mse:.4f}")

difference = true_mse - mse
relative_error = abs(difference) / mse * 100

print(f"Difference       : {difference:.4f}")
print(f"Relative Error   : {relative_error:.4f}%")
print(f"MSE Similarity   : {(100 - relative_error):.2f}%")