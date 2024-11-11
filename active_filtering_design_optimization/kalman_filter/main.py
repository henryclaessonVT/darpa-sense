import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.x = np.zeros((state_dim, 1))  # Initial state
        self.P = np.eye(state_dim)          # State covariance
        self.Q = np.eye(state_dim) * 0.1    # Process noise covariance
        self.R = np.eye(measurement_dim) * 0.5  # Measurement noise covariance
        self.F = np.eye(state_dim)          # State transition matrix
        self.H = np.eye(measurement_dim, state_dim)  # Measurement matrix

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(-1, 1)  # Ensure z is a column vector
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x += K @ y  # Update the state estimate
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # Update the error covariance


# Example usage:
state_dim = 4  # Thermal gradients across 4 corners
measurement_dim = 4

kf = KalmanFilter(state_dim, measurement_dim)

# Number of iterations (time steps)
num_iterations = 50

# Simulated true temperatures over time for each corner
true_temperatures = np.array([[20 + np.sin(0.1 * t) * 5 for t in range(num_iterations)],
                               [15 + np.sin(0.1 * t + 1) * 5 for t in range(num_iterations)],
                               [18 + np.sin(0.1 * t + 2) * 5 for t in range(num_iterations)],
                               [17 + np.sin(0.1 * t + 3) * 5 for t in range(num_iterations)]]).T

# Add some noise to simulate measurements
measurement_noise = np.random.normal(0, 1, true_temperatures.shape)
measurements = true_temperatures + measurement_noise

# Lists to store estimates and measurements for plotting
estimates = []
measured_values = []

# Run the Kalman filter for the specified number of iterations
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    estimates.append(kf.x.flatten())
    measured_values.append(measurement.flatten())

# Convert lists to numpy arrays for easier plotting
estimates = np.array(estimates)
measured_values = np.array(measured_values)

# Plotting
plt.figure(figsize=(12, 8))
for i in range(state_dim):
    plt.subplot(state_dim, 1, i + 1)
    plt.plot(estimates[:, i], label='Estimated', marker='o')
    plt.plot(measured_values[:, i], label='Measured', linestyle='--', marker='x')
    plt.title(f'Thermal Gradient at Corner {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
