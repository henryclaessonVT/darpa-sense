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

# Function to read FEA results from CSV file
def read_fea_results(filename):
    positions = []
    fea_strains = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        for row in reader:
            positions.append(float(row[0]))  # First column: positions
            fea_strains.append([float(strain) for strain in row[1:]])  # Strain values for each load case
    return np.array(positions), np.array(fea_strains)

# Function to find closest FEA result
def find_closest_fea(fea_strains, measured_strains, sensor_indices):
    min_diff = np.inf
    closest_fea_idx = -1
    for i in range(fea_strains.shape[1]):  # Loop through each load strain valcase (columns)
        fea_values = fea_strains[sensor_indices, i]  # FEA ues at sensor locations
        diff = np.linalg.norm(fea_values - measured_strains)  # Calculate Euclidean distance
        if diff < min_diff:
            min_diff = diff
            closest_fea_idx = i
    return closest_fea_idx

# Function to run alpha-beta-gamma filtering with sensor data
def run_alpha_beta_gamma_filter(fea_strains, sensor_data, sensor_indices, delta_t, alpha=0.25, beta=0.05, gamma=0.0001):
    # Initialize filter for each sensor
    filters = [AlphaBetaGammaFilter(alpha, beta, gamma) for _ in sensor_indices]
    
    filtered_strains = []
    
    for time_step, measured_strains in enumerate(sensor_data):  # Loop through time steps
        updated_strains = []
        for i, sensor_index in enumerate(sensor_indices):
            # Update filter for each sensor
            est_strain, est_rate, est_acceleration = filters[i].update(measured_strains[i], delta_t)
            updated_strains.append(est_strain)
        
        # Find the closest FEA model based on updated strain estimates
        closest_fea_idx = find_closest_fea(fea_strains, updated_strains, sensor_indices)
        
        print(f"Time step {time_step}: Closest FEA model is load case {closest_fea_idx} with updated strains {updated_strains}")

        filtered_strains.append(updated_strains)
    
    return filtered_strains


# Function to plot filtered strain estimates vs sensor data
def plot_filter_convergence(time_steps, sensor_data, filtered_strains, sensor_indices):
    plt.figure(figsize=(10, 6))
    
    for i, sensor_index in enumerate(sensor_indices):
        plt.plot(time_steps, [sensor_data[t][i] for t in time_steps], label=f"Sensor Data (Element {sensor_index})", linestyle='--')
        plt.plot(time_steps, [filtered_strains[t][i] for t in time_steps], label=f"Filtered Strain (Element {sensor_index})", marker='o')

    plt.title("Filter Convergence: Estimated Strains vs Measured Strains")
    plt.xlabel("Time Step")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot error between filtered strain estimates and sensor data
def plot_filter_error(time_steps, sensor_data, filtered_strains, sensor_indices):
    plt.figure(figsize=(10, 6))

    for i, sensor_index in enumerate(sensor_indices):
        errors = [abs(sensor_data[t][i] - filtered_strains[t][i]) for t in time_steps]
        plt.plot(time_steps, errors, label=f"Error (Element {sensor_index})", marker='o')

    plt.title("Filter Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to compare the FEA strain results with the filter's estimated strain
def plot_fea_comparison(fea_strains, closest_fea_idx, filtered_strains, sensor_indices, time_step):
    plt.figure(figsize=(10, 6))
    
    fea_strains_at_sensors = fea_strains[sensor_indices, closest_fea_idx]  # FEA strains at sensor locations for closest load case
    filtered_strain_values = filtered_strains[time_step]
    
    plt.bar([f"Element {sensor}" for sensor in sensor_indices], fea_strains_at_sensors, label="FEA Strain (Closest Model)", alpha=0.6)
    plt.bar([f"Element {sensor}" for sensor in sensor_indices], filtered_strain_values, label="Filtered Strain", alpha=0.6)

    plt.title(f"Comparison at Time Step {time_step} with Closest FEA Model (Load Case {closest_fea_idx})")
    plt.ylabel("Strain")
    plt.legend()
    plt.show()

def plot_all_fea_results(positions, strains):
    n = len(strains)
    plt.figure(figsize=(6, 4))
    loads = np.linspace(100, 2000, 15)
    for i in range(n):
        print(strains[:,i])
        plt.scatter(positions, strains[:,i], label=f'Load = {np.floor(loads[i])} N') 
        plt.plot(positions, strains[:,i], linestyle='--')

    plt.title(f'Strain Distribution for Load', fontweight='bold')# = {load} N')
    plt.xlabel('Position along Beam (m)', fontsize=20)
    plt.ylabel('Strain', fontsize=20)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return

filename = "beam_strain_data.csv"  # The CSV file containing FEA results
positions, fea_strains = read_fea_results(filename)
random_seed = np.random.seed(50)
true_load_case = 7
duration = 60
ground_truth = fea_strains[:,true_load_case]


strain_gauge_data = []
for _ in range(duration):
    noisy_data = ground_truth #+ np.random.normal(0, ground_truth * .02, ground_truth.shape)
    strain_gauge_data.append(noisy_data)





sensor_indices = [3, 5]  # Sensors placed at element 3 and 5
delta_t = 1.0  # Time step in seconds
# print(strain_gauge_data[sensor_indices])


# Run alpha-beta-gamma filter
filtered_strains = run_alpha_beta_gamma_filter(fea_strains, strain_gauge_data, sensor_indices, delta_t)

# Example usage with your filtered strain data and sensor data
time_steps = np.arange(len(strain_gauge_data))  # Assuming sensor_data is a list of time step data


# # Plot filter convergence (how the estimated strain matches sensor data over time)
plot_filter_convergence(time_steps, strain_gauge_data, filtered_strains, sensor_indices)

# # Plot filter error over time
plot_filter_error(time_steps, strain_gauge_data, filtered_strains, sensor_indices)

# # Compare filter's estimated strain with the closest FEA result for a specific time step (e.g., time_step=2)
time_step_to_plot = 59
closest_fea_idx = find_closest_fea(fea_strains, filtered_strains[time_step_to_plot], sensor_indices)
plot_fea_comparison(fea_strains, closest_fea_idx, filtered_strains, sensor_indices, time_step_to_plot)

# uncomment to see all loads
# plot_all_fea_results(positions, fea_strains)
