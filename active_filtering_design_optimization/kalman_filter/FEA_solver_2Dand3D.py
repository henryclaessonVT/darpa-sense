import numpy as np
import matplotlib.pyplot as plt
import csv

# Material properties for Aluminum
E = 69e9  # Young's modulus in Pa
nu = 0.33  # Poisson's ratio

# Geometry dimensions
length = 0.1  # Length in meters (10 cm)
width = 0.02  # Width in meters (2 cm)
height = 0.005  # Height in meters (0.5 cm)

# Function to calculate 2D FEA data for a beam
def calculate_2d_fea(load_cases, num_elements_x=10, num_elements_y=5):
    # Create a grid of nodes
    x = np.linspace(0, length, num_elements_x)
    y = np.linspace(0, width, num_elements_y)
    strains = np.zeros((num_elements_x, num_elements_y, len(load_cases)))

    # Loop over each load case
    for i, load in enumerate(load_cases):
        for j in range(num_elements_x):
            for k in range(num_elements_y):
                # Simple beam theory for bending (simplified for demonstration)
                if j == num_elements_x - 1:  # Free end
                    strains[j, k, i] = (load * (length - x[j]) * width ** 2) / (2 * E * height)  # Bending strain

    return strains, x, y

# Function to calculate 3D FEA data for a beam
def calculate_3d_fea(load_cases, num_elements_x=10, num_elements_y=5, num_elements_z=2):
    # Create a grid of nodes
    x = np.linspace(0, length, num_elements_x)
    y = np.linspace(0, width, num_elements_y)
    z = np.linspace(0, height, num_elements_z)
    strains = np.zeros((num_elements_x, num_elements_y, num_elements_z, len(load_cases)))

    # Loop over each load case
    for i, load in enumerate(load_cases):
        for j in range(num_elements_x):
            for k in range(num_elements_y):
                for l in range(num_elements_z):
                    # Simple beam theory for bending (simplified for demonstration)
                    if j == num_elements_x - 1 and l == 0:  # Free end, bottom layer
                        strains[j, k, l, i] = (load * (length - x[j]) * width ** 2) / (2 * E * height)  # Bending strain

    return strains, x, y, z

# Function to export FEA results to a CSV file
def export_fea_results(filename, strains, x, y, z=None):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if z is None:  # 2D case
            writer.writerow(['X', 'Y'] + [f'Strain_Load_{i+1}' for i in range(strains.shape[2])])
            for i in range(strains.shape[0]):
                for j in range(strains.shape[1]):
                    row = [x[i], y[j]] + strains[i, j, :].tolist()
                    writer.writerow(row)
        else:  # 3D case
            writer.writerow(['X', 'Y', 'Z'] + [f'Strain_Load_{i+1}' for i in range(strains.shape[3])])
            for i in range(strains.shape[0]):
                for j in range(strains.shape[1]):
                    for k in range(strains.shape[2]):
                        row = [x[i], y[j], z[k]] + strains[i, j, k, :].tolist()
                        writer.writerow(row)

# Define load cases
load_cases = [1000, 10000, 20000]  # Load cases in Newtons

# Generate FEA data
strains_2d, x_2d, y_2d = calculate_2d_fea(load_cases)
strains_3d, x_3d, y_3d, z_3d = calculate_3d_fea(load_cases)

# Export results to CSV files
export_fea_results("2d_fea_results.csv", strains_2d, x_2d, y_2d)
export_fea_results("3d_fea_results.csv", strains_3d, x_3d, y_3d, z_3d)

# Example plotting for 2D strain field
plt.figure()
plt.imshow(strains_2d[:, :, -1], extent=[0, length, 0, width], origin='lower', cmap='viridis')
plt.colorbar(label='Strain')
plt.title('2D Strain Field for Load Case 3 (2000 N)')
plt.xlabel('Length (m)')
plt.ylabel('Width (m)')
plt.show()
