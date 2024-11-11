import numpy as np
import matplotlib.pyplot as plt
import csv

# Material properties and beam parameters
E = 70e9  # Young's modulus for Aluminum (in Pascals)
I = 1e-8  # Moment of inertia (m^4), assume for now
L_total = 0.1  # Total length of beam (10 cm)
num_elements = 10
element_length = L_total / num_elements

# Function to create the element stiffness matrix for bending
def element_stiffness_matrix_bending(E, I, L):
    return (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])

# Assemble global stiffness matrix
def global_stiffness_matrix_bending(num_elements, E, I, L):
    K = np.zeros((2*num_elements+2, 2*num_elements+2))
    k_e = element_stiffness_matrix_bending(E, I, L)
    for i in range(num_elements):
        K[2*i:2*i+4, 2*i:2*i+4] += k_e
    return K

# Apply boundary conditions and solve for displacements
def solve_displacements(K, F, fixed_nodes):
    # Remove the rows and columns corresponding to the fixed nodes
    for node in sorted(fixed_nodes, reverse=True):
        K = np.delete(np.delete(K, node, axis=0), node, axis=1)
        F = np.delete(F, node)
    
    # Solve for displacements
    displacements = np.linalg.solve(K, F)
    
    # Reinsert the fixed node displacement (which is zero)
    full_displacements = np.zeros(K.shape[0] + len(fixed_nodes))
    full_displacements[np.setdiff1d(np.arange(K.shape[0] + len(fixed_nodes)), fixed_nodes)] = displacements
    
    return full_displacements

# Calculate strains
def calculate_strains_bending(displacements, L):
    strains = []
    for i in range(num_elements):
        strain = (displacements[2*i+2] - displacements[2*i]) / L
        strains.append(strain)
    return np.array(strains)

# Plot Heatmap
def plot_heatmap(positions, strains, load):
    plt.figure(figsize=(6, 4))
    plt.imshow([strains], cmap='viridis', aspect='auto', extent=[positions[0], positions[-1], 0, 1])
    plt.colorbar(label='Strain')
    plt.title(f'Strain Heatmap for Load = {load} N')
    plt.xlabel('Position along Beam (m)')
    plt.ylabel('Strain (Normalized)')
    plt.show()

# Plot XY scatter plot of position vs strain
def plot_strain_scatter(positions, strains, load):
    plt.figure(figsize=(6, 4))
    plt.scatter(positions, strains, color='r', label=f'Load = {load} N')
    plt.plot(positions, strains, linestyle='--', color='b')
    plt.title(f'Strain Distribution for Load = {load} N')
    plt.xlabel('Position along Beam (m)')
    plt.ylabel('Strain')
    plt.grid(True)
    plt.legend()
    plt.show()

# Function to export results to CSV
def export_to_csv(positions, strains, loads, filename="strain_data.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ['Position (m)'] + [f'Strain (Load = {load} N)' for load in loads]
        writer.writerow(header)
        
        # Write data for each position
        for i in range(len(positions)):
            row = [positions[i]] + [strain[i] for strain in strains]
            writer.writerow(row)
    print(f"Data successfully exported to {filename}")

# Run FEA for different load cases
def run_fea_bending(loads, export=False, filename="strain_data.csv"):
    # Initialize the stiffness matrix
    K = global_stiffness_matrix_bending(num_elements, E, I, element_length)
    
    # Get positions of each element's midpoint
    positions = np.linspace(0, L_total, num_elements+1)
    
    # Store strain data for each load
    all_strains = []
    
    # Loop through the load cases
    for load in loads:
        # Initialize the force vector
        F = np.zeros(2*num_elements + 2)
        F[-2] = load  # Apply load at the free end (force applied at last displacement dof)
        
        # Solve for displacements
        displacements = solve_displacements(K, F, fixed_nodes=[0, 1])  # Fix both displacement and rotation at node 0
        
        # Calculate strains for each element
        strains = calculate_strains_bending(displacements, element_length)
        all_strains.append(strains)
        
        # Plot the heatmap and scatter plot
        plot_heatmap(positions[:-1], strains, load)  # Heatmap for strain distribution
        plot_strain_scatter(positions[:-1], strains, load)  # XY scatter plot

    # Export data to CSV if needed
    if export:
        export_to_csv(positions[:-1], all_strains, loads, filename)

# Example load cases
# loads = [100, 500, 1000]  # Loads in Newtons
loads = np.linspace(100, 2000, 15)

# Run the simulation, plot results, and export to CSV
run_fea_bending(loads, export=True, filename="beam_strain_data.csv")
