import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_vtu_file(file_path):
    # Read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    # Get the output data (vtkUnstructuredGrid)
    unstructured_grid = reader.GetOutput()
    # Extract point coordinates (nodes)
    points = unstructured_grid.GetPoints()
    point_coords = vtk_to_numpy(points.GetData())
    # Extract any result data arrays (e.g., stress, strain)
    point_data = unstructured_grid.GetPointData()
    # Assuming you want to extract the first array (adjust as needed)
    result_array = vtk_to_numpy(point_data.GetArray(0)) if point_data.GetNumberOfArrays() > 0 else None
    return point_coords, result_array

def plot_vtu_data(point_coords, result_array=None, xlim=None, ylim=None, zlim=None):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points (nodes)
    ax.scatter(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2], c='b', marker='o')
    # If result data (e.g., stress, strain) exists, use it for color coding
    if result_array is not None:
        # Normalize result array for color scaling
        norm = plt.Normalize(np.min(result_array), np.max(result_array))
        ax.scatter(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2], c=result_array, cmap='turbo', marker='o')
        fig.colorbar(ax.scatter(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2], c=result_array, cmap='turbo'), ax=ax, label="Strain")
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.title('VTU Data Visualization')
    plt.legend()
    plt.show()

def plot_surface_cross_section(point_coords, result_array, y_center, y_thickness=2, xlim=None, ylim=None, zlim=None):
    # Filter points within the 2mm thick slice along Y axis
    half_thickness = y_thickness / 2
    mask = np.abs(point_coords[:, 1] - y_center) <= half_thickness
    x_filtered = point_coords[mask][:, 0]
    z_filtered = point_coords[mask][:, 2]
    strain_filtered = result_array[mask]
    # Create a grid based on the X and Z points
    grid_resolution = 100  # Number of bins in X and Z direction
    x_min, x_max = np.min(x_filtered), np.max(x_filtered)
    z_min, z_max = np.min(z_filtered), np.max(z_filtered)
    # Create a 2D histogram grid of strain values
    grid, x_edges, z_edges = np.histogram2d(x_filtered, z_filtered, bins=grid_resolution, weights=strain_filtered)
    counts, _, _ = np.histogram2d(x_filtered, z_filtered, bins=grid_resolution)  # Count number of points in each bin
    # Avoid division by zero for empty bins
    grid = np.divide(grid, counts, where=(counts != 0))
    # Create meshgrid for X, Z values
    X, Z = np.meshgrid(x_edges[:-1], z_edges[:-1])
    # Plot as a 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create a surface plot using X, Z, and the grid of strain values as the height
    surf = ax.plot_surface(X, Z, grid.T, cmap='turbo')
    # Add color bar and labels
    fig.colorbar(surf, ax=ax, label="Strain")
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Z Position (mm)')
    ax.set_zlabel('Strain')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.title(f"Strain Cross-Section Surface at Y = {y_center}mm")
    plt.show()
    return X, Z, grid.T


def export_feild_to_csv(X, Z, strain, f):

    for idx, row in enumerate(X):
        if idx == 1:
            print('Row 1 Starts hers')
            print(row)
            print('\n\n')
            for pos in row:
                print(pos)


    print(np.shape(X))
    print(np.shape(Z))
    print(np.shape(strain))

    red = []
    for idxr, row in enumerate(X):
        for idxc, col in enumerate(row):
            red.append([X[idxr][idxc], Z[idxr][idxc], strain[idxr][idxc]])
    
    print(red)



    with open(f, 'w') as file:
        for item in red:
            exportable = str(item[0])+','+str(item[1])+','+str(item[2])+'\n'
            file.writelines(exportable)
    return



if __name__ == "__main__":
    # Replace with the path to your VTU file
    file_path = 'FEAComparisonCode\\results3.vtu'
    # Define limits for x, y, and z axes (adjust as needed)
    x_limits = (-200, 200)  # Example: set X axis limits from -10 to 10
    y_limits = (-200, 200)  # Example: set Y axis limits from -10 to 10
    z_limits = (-200, 200)    # Example: set Z axis limits from -5 to 5
    # Read the VTU file
    point_coords, result_array = read_vtu_file(file_path)

    # Plot the data
    plot_vtu_data(point_coords, result_array, xlim=x_limits, ylim=y_limits, zlim=z_limits)

    # Define the Y plane for the cross-section and the thickness (2mm)
    y_center = 0  # Cross-section at Y = 0mm (adjust as needed)
    y_thickness = 2  # 2mm thick slice along Y axis
    z_limits = (-.000075, .0001)    # Example: set Z axis limits from -5 to 5
    # Plot the cross-section along the X-Z plane with strain data
    # Plot the surface cross-section along the X-Z plane with strain data
    X, Z, e = plot_surface_cross_section(point_coords, result_array, y_center, y_thickness, xlim=x_limits, ylim=y_limits, zlim=z_limits)

    fileout = 'output2.csv'
    
    export_feild_to_csv(X, Z, e, fileout)




