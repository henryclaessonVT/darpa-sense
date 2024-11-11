import numpy as np
import csv
import matplotlib.pyplot as plt

# Alpha-Beta-Gamma Filter class
class AlphaBetaGammaFilter:
    def __init__(self, alpha, beta, gamma, initial_strain=0.0, initial_rate=0.0, initial_acceleration=0.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.strain_est = initial_strain
        self.rate_est = initial_rate
        self.acceleration_est = initial_acceleration

    def update(self, measured_strain, delta_t):
        # Predict the next strain state
        predicted_strain = self.strain_est + self.rate_est * delta_t + 0.5 * self.acceleration_est * delta_t**2
        predicted_rate = self.rate_est + self.acceleration_est * delta_t
        
        # Calculate residual (difference between measured and predicted strain)
        residual = measured_strain - predicted_strain
        
        # Update estimates using alpha, beta, and gamma gains
        self.strain_est += self.alpha * residual
        self.rate_est += self.beta * residual / delta_t
        self.acceleration_est += self.gamma * residual / (0.5 * delta_t**2)
        
        return self.strain_est, self.rate_est, self.acceleration_est

# Function to read FEA results for a 2D case from a CSV file
def read_fea_results_2d(filename):
    positions = []
    fea_strains = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        for row in reader:
            # Assuming CSV contains x, y, and strain values for multiple load cases
            x, y = float(row[0]), float(row[1])
            positions.append([x, y])
            fea_strains.append([float(strain) for strain in row[2:]])  # Strain values for each load case
    return np.array(positions), np.array(fea_strains)

# Function to find closest FEA result (2D case)
def find_closest_fea_2d(fea_strains, measured_strains, sensor_indices):
    min_diff = np.inf
    closest_fea_idx = -1
    for i in range(fea_strains.shape[1]):  # Loop through each load case (columns)
        fea_values = fea_strains[sensor_indices, i]  # FEA strain values at sensor locations
        diff = np.linalg.norm(fea_values - measured_strains)  # Euclidean distance
        if diff < min_diff:
            min_diff = diff
            closest_fea_idx = i
    return closest_fea_idx

# Function to run alpha-beta-gamma filtering with sensor data (2D case)
def run_alpha_beta_gamma_filter_2d(fea_strains, sensor_data, sensor_indices, delta_t, alpha=0.85, beta=0.005, gamma=0.0001):
    filters = [AlphaBetaGammaFilter(alpha, beta, gamma) for _ in sensor_indices]
    filtered_strains = []

    for time_step, measured_strains in enumerate(sensor_data):  # Loop through time steps
        updated_strains = []
        for i, sensor_index in enumerate(sensor_indices):
            # Update filter for each sensor
            est_strain, est_rate, est_acceleration = filters[i].update(measured_strains[i], delta_t)
            updated_strains.append(est_strain)
        
        # Find the closest FEA model based on updated strain estimates
        closest_fea_idx = find_closest_fea_2d(fea_strains, updated_strains, sensor_indices)
        print(f"Time step {time_step}: Closest FEA model is load case {closest_fea_idx} with updated strains {updated_strains}")
        filtered_strains.append(updated_strains)
    
    return filtered_strains

# Plotting for the 2D case (using a heatmap for strain field visualization)
def plot_2d_heatmap(fea_strains, sensor_data, positions, closest_fea_idx):
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    strains = fea_strains[:, closest_fea_idx]

    plt.scatter(x_coords, y_coords, c=strains, cmap='viridis', marker='s')
    plt.colorbar(label="Strain")
    plt.title(f"2D Strain Field for Load Case {closest_fea_idx}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

# Example usage for the 2D case
filename_2d = "beam_strain_data_2d.csv"  # The CSV file containing FEA results for 2D
positions_2d, fea_strains_2d = read_fea_results_2d(filename_2d)

# Example sensor data (strain measurements at different time steps for elements)
sensor_data_2d = [
    [0.0025, 0.0018],  # Time step 1 strain values at sensor locations
    [0.0026, 0.0019],  # Time step 2 strain values
    [0.0027, 0.0020],  # Time step 3 strain values
]

sensor_indices_2d = [3, 5]  # Sensors placed at specific (x, y) elements
delta_t_2d = 1.0  # Time step in seconds

# Run alpha-beta-gamma filter for 2D case
filtered_strains_2d = run_alpha_beta_gamma_filter_2d(fea_strains_2d, sensor_data_2d, sensor_indices_2d, delta_t_2d)

# Plot the 2D heatmap for the closest FEA model
closest_fea_idx_2d = find_closest_fea_2d(fea_strains_2d, filtered_strains_2d[-1], sensor_indices_2d)
plot_2d_heatmap(fea_strains_2d, sensor_data_2d, positions_2d, closest_fea_idx_2d)
