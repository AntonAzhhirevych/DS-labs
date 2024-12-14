import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter

# Step 1: Generate synthetic input data with anomalies
def generate_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x**2 - 3 * x + 5 + np.random.normal(0, 5, size=x.shape)
    y[::10] += 50  # Inject anomalies
    return x, y

# Step 2: Detect and remove anomalies using Z-score method
def remove_anomalies(x, y):
    z_scores = (y - np.mean(y)) / np.std(y)
    non_anomalous_indices = np.abs(z_scores) < 3
    return x[non_anomalous_indices], y[non_anomalous_indices]

# Step 3: Fit polynomial regression model
def fit_polynomial_regression(x, y, degree=2):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)
    return model, y_pred, poly

# Step 4: Extrapolate future values
def extrapolate(model, poly, x, interval_ratio=0.5):
    interval = (x.max() - x.min()) * interval_ratio
    x_future = np.linspace(x.min(), x.max() + interval, 100)
    x_future_poly = poly.transform(x_future.reshape(-1, 1))
    y_future = model.predict(x_future_poly)
    return x_future, y_future

# Step 5: Apply alpha-beta filter for smoothing
def alpha_beta_filter(data, alpha=0.85, beta=0.005):
    smoothed = []
    estimate = data[0]
    trend = 0
    for i in range(1, len(data)):
        prev_estimate = estimate
        estimate += trend
        residual = data[i] - estimate
        estimate += alpha * residual
        trend += beta * residual
        smoothed.append(estimate)
    return np.array(smoothed)




# Main execution
if __name__ == "__main__":
    # Step 1: Generate data
    x, y = generate_data()

    # Plot 1: Original Data with Anomalies
    plt.figure()
    plt.scatter(x, y, label="Original Data with Anomalies", color="red", alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Original Data with Anomalies")
    plt.legend()
    plt.savefig("1_original_data.png")
    plt.show()

    # Step 2: Remove anomalies
    x_clean, y_clean = remove_anomalies(x, y)

    # Plot 2: Cleaned Data
    plt.figure()
    plt.scatter(x_clean, y_clean, label="Cleaned Data", color="blue")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cleaned Data")
    plt.legend()
    plt.savefig("2_cleaned_data.png")
    plt.show()

    # Step 3: Polynomial regression
    model, y_pred, poly = fit_polynomial_regression(x_clean, y_clean)

    # Plot 3: Polynomial Regression
    plt.figure()
    plt.scatter(x_clean, y_clean, label="Cleaned Data", color="blue")
    plt.plot(x_clean, y_pred, label="Polynomial Regression", color="green")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polynomial Regression")
    plt.legend()
    plt.savefig("3_polynomial_regression.png")
    plt.show()

    # Step 4: Extrapolation
    x_future, y_future = extrapolate(model, poly, x_clean)

    # Plot 4: Extrapolated Data
    plt.figure()
    plt.scatter(x_clean, y_clean, label="Cleaned Data", color="blue")
    plt.plot(x_future, y_future, label="Extrapolated Data", linestyle="--", color="orange")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Extrapolated Data")
    plt.legend()
    plt.savefig("4_extrapolated_data.png")
    plt.show()

    # Step 5: Alpha-beta filtering
    y_smoothed = alpha_beta_filter(y_future)

    # Plot 5: Smoothed Data
    plt.figure()
    plt.plot(x_future, y_future, label="Extrapolated Data", linestyle="--", color="orange")
    plt.plot(x_future[1:], y_smoothed, label="Smoothed Data", linestyle="-.", color="purple")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Smoothed Data")
    plt.legend()
    plt.savefig("5_smoothed_data.png")
    plt.show()

    # Analysis
    mse = mean_squared_error(y_clean, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print("Script completed successfully!")


    # if __name__ == "__main__":
    # # Step 1: Generate data
    # x, y = generate_data()

    # # Step 2: Remove anomalies
    # x_clean, y_clean = remove_anomalies(x, y)

    # # Step 3: Polynomial regression
    # model, y_pred, poly = fit_polynomial_regression(x_clean, y_clean)

    # # Step 4: Extrapolation
    # x_future, y_future = extrapolate(model, poly, x_clean)

    # # Step 5: Alpha-beta filtering
    # y_smoothed = alpha_beta_filter(y_future)

    # # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x, y, label="Original Data with Anomalies", color="red", alpha=0.6)
    # plt.scatter(x_clean, y_clean, label="Cleaned Data", color="blue")
    # plt.plot(x_clean, y_pred, label="Polynomial Regression", color="green")
    # plt.plot(x_future, y_future, label="Extrapolated Data", linestyle="--", color="orange")
    # plt.plot(x_future[1:], y_smoothed, label="Smoothed Data", linestyle="-.", color="purple")
    # plt.legend()
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Data Cleaning, Polynomial Regression, and Extrapolation")
    # plt.show()


  
