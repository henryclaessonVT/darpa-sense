import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Alpha-Beta-Gamma filter class
class AlphaBetaGammaFilter:
    def __init__(self, alpha, beta, gamma, dt):
        # Filter parameters
        self.alpha = alpha  # Position/strain adjustment
        self.beta = beta    # Velocity/load adjustment
        self.gamma = gamma  # Acceleration/displacement adjustment
        self.dt = dt        # Time step

        # Initial states: [strain, load (velocity), displacement (acceleration)]
        self.state = np.array([0.0, 0.0, 0.0])  # [strain, load, displacement]
        self.estimated_strain = []  # To track estimates over time
    
    def predict(self):
        # Predict next state (basic model: constant velocity & acceleration over time)
        strain, load, displacement = self.state
        strain = strain + load * self.dt + 0.5 * displacement * self.dt ** 2
        load = load + displacement * self.dt
        return np.array([strain, load, displacement])
    
    def update(self, measured_strain):
        # Update state using measurements and the alpha-beta-gamma filter equations
        predicted_state = self.predict()
        predicted_strain = predicted_state[0]

        # Error between measured strain and predicted strain
        error = measured_strain - predicted_strain

        # Update estimates with alpha, beta, and gamma
        self.state[0] = predicted_state[0] + self.alpha * error  # Update strain
        self.state[1] = predicted_state[1] + self.beta * error / self.dt  # Update load
        self.state[2] = predicted_state[2] + self.gamma * error / (0.5 * self.dt ** 2)  # Update displacement

        # Store estimated strain for tracking
        self.estimated_strain.append(self.state[0])
    
    def get_estimated_strain(self):
        return self.estimated_strain

# Simulate strain gauge data (example)
def simulate_strain_gauges(timesteps):
    # Assume some true strain values over time (can replace with actual data)
    true_strain = 0.05 * np.sin(0.2 * np.pi * timesteps) + 0.01 * timesteps
    return true_strain

# Interpolate strain between gauge points
def interpolate_strain(gauge_positions, gauge_strains, interpolate_positions):
    interpolator = interp1d(gauge_positions, gauge_strains, kind='cubic')
    interpolated_strain = interpolator(interpolate_positions)
    return interpolated_strain

# Example usage
if __name__ == "__main__":
    # Time step and number of iterations
    dt = 0.1
    num_steps = 10000
    timesteps = np.arange(0, num_steps * dt, dt)

    # Simulate strain gauge data
    gauge_strain_data = simulate_strain_gauges(timesteps)

    # Alpha-Beta-Gamma filter initialization (tune these values)
    alpha = 0.85
    beta = 0.005
    gamma = 0.0001
    filter = AlphaBetaGammaFilter(alpha, beta, gamma, dt)

    # Run filter over time steps
    for t in range(num_steps):
        measured_strain = gauge_strain_data[t]  # Simulated strain gauge reading
        filter.update(measured_strain)

    # Get estimated strain over time
    estimated_strain = filter.get_estimated_strain()

    # Interpolating strain at points between the strain gauges
    gauge_positions = np.array([0, 1, 2, 3])  # Example positions of strain gauges
    gauge_strains = [estimated_strain[-1]] * len(gauge_positions)  # Last estimated strain for simplicity
    interpolate_positions = np.linspace(0, 3, 50)
    interpolated_strain = interpolate_strain(gauge_positions, gauge_strains, interpolate_positions)

    # Plot results
    plt.figure()
    plt.plot(timesteps, gauge_strain_data, label='True Strain')
    plt.plot(timesteps, estimated_strain, label='Estimated Strain', linestyle='--')
    plt.title('Strain vs Time')
    plt.xlabel('Time')
    plt.ylabel('Strain')
    plt.legend()
    plt.show()

    # Plot interpolated strain
    plt.figure()
    plt.plot(interpolate_positions, interpolated_strain, label='Interpolated Strain')
    plt.title('Interpolated Strain Between Gauge Points')
    plt.xlabel('Position along bar')
    plt.ylabel('Strain')
    plt.legend()
    plt.show()
