import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_raw_CSV(file_path):
    '''
  Reads a csv file, extracts data.

  Args:
      filename: Path to the csv file.
  Returns:
      array for time, temp1, temp2, temp3, temp4 and raw ADC strain
    '''
    # initialize the output arrays
    arr = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        # loop through each line and append each output array to have the appropriate data
        
        for line in lines:
            x = line.split(",")
            inter = []
            for item in x:
                inter.append(float(item))
            arr.append(inter)
    return arr

def sort_CSV_data(arr):

    groups = []
    count = [1.0, 2.0, 3.0, 4.0]
    intgroups = []
    for item in arr:
        if count == []:
            count = [1.0, 2.0, 3.0, 4.0]
            groups.append(intgroups)
            intgroups = []
        if item[1] in count:
            intgroups.append(item)
            count.remove(item[1])
    return groups

def find_highest_correlation(original, dataset):
    highest_corr = -1  # Start with a low correlation value
    best_match = None
    best_index = -1
    
    for i, array in enumerate(dataset):
        # Calculate Pearson correlation coefficient
        corr_coeff = np.corrcoef(original, array)[0, 1]
        
        if corr_coeff > highest_corr:
            highest_corr = corr_coeff
            best_match = array
            best_index = i  # Track the index of the array with the highest correlation
    
    return best_match, highest_corr, best_index

def conv_adc_to_strain(adc_values):

    Rg = 350
    Vin = 5.0
    balanced_adc_val = 512
    gauge_factor = 2.0
    strain_vec = []
    for value in adc_values:
        red = (value/balanced_adc_val)*Vin
        temp_strain_val = gauge_factor*(red/Vin)
        strain_vec.append(temp_strain_val)
    return strain_vec

def sort_strain_data_by_idx(arr, locations):
    outarr = []
    for timestep in arr:
        tempx = []
        tempy = []
        for sensorReading in timestep:
            ID = sensorReading[1]
            (x, y) = locations[ID]
            xpos = x
            ypos = y
            tempx.append(y)
            tempy.append(sensorReading[2])
        outarr.append([tempx, tempy])
    return outarr

def sort_temp_data_by_idx(arr, locations):
    outarr = []
    for timestep in arr:
        tempx = []
        tempy = []
        for sensorReading in timestep:
            ID = sensorReading[1]
            (x, y) = locations[ID]
            xpos = x
            ypos = y
            tempx.append(y)
            tempy.append(sensorReading[3])
        outarr.append([tempx, tempy])
    return outarr

def zero_mean_each_dataset(arr):
    zero_mean_arr = {1: 25, 2: -7, 3: -12, 4: 24}
    result_dict = {}
    # print(arr[0])

    for key, value in zero_mean_arr.items():
        idx = arr[0].index(key-1)
        arr[1][idx] = arr[1][idx] - value
        


    # Iterate through the list of IDs and values

    return arr


filepath = 'FEAComparisonCode\\demo_results.csv'
f = 'FEAComparisonCode\\output2.csv'

# import CSV data from experiment, reorder and group the strainfeild from each 
results = read_raw_CSV(filepath)
# sort the experimental data by time stamp
groups = sort_CSV_data(results) # output is in the format [[timestamp1 set of four points], [timestamp1 set of four points], ...]


# import data from FEA rsults (this comes from the visualize.py script)
data = read_raw_CSV(f)
# print(data)
# Desired Z value and margin
target_Z = 12.7
margin = 0.2
filtered_data = [arr for arr in data if target_Z - margin <= arr[1] <= target_Z + margin]

# [[time, ID, strain, temp], [time, ID, strain, temp], [time, ID, strain, temp], [time, ID, strain, temp]]
# |                          |                      |                                                    |                                                                             
# |                          |                      |                                                    |
# |                          |______________________| <--- SensorReading                                 |
# |______________________________________________________________________________________________________|<--- TimeStamp



locations = {3: (0, 0), 1: (0, 1), 4: (0, 2), 2: (0, 3)} # from experiment

comparabledata = sort_strain_data_by_idx(groups, locations) # converts each timestep to a list of indices and strain values in the format:4
comparabledatatemp = sort_temp_data_by_idx(groups, locations) # converts each timestep to a list of indices and strain values in the format:

print(comparabledatatemp)


# Loop through each pair of index and temperature arrays
sorted_temps = []
for pair in comparabledatatemp:
    indices, temps = pair
    
    # Zip index and temperature arrays together
    zipped = list(zip(indices, temps))
    
    # Sort by index
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    
    # Extract sorted temperatures
    sorted_temp = [temp for _, temp in sorted_zipped]
    
    # Append to the sorted_temps list
    sorted_temps.append(sorted_temp)

print(sorted_temps)

temp1 = []
temp2 = []
temp3 = []
temp4 = []

for item in sorted_temps:
    temp1.append(item[0])
    temp2.append(item[1])
    temp3.append(item[2])
    temp4.append(item[3])


# locations = {3: (0, 0), 1: (0, 1), 4: (0, 2), 2: (0, 3)} # from experiment


plt.style.use('seaborn-darkgrid')
xdata = range(len(temp1))
plt.plot(xdata, temp4, color='y', label='Strain from Sensor 4')
plt.plot(xdata, temp1, color='g', label='Strain from Sensor 1')
plt.plot(xdata, temp2, color='r', label='Strain from Sensor 2')
plt.plot(xdata, temp3, color='b', label='Strain from Sensor 3')
plt.xlim([0, 300])

plt.legend(loc='upper right', fontsize=12)

# Title with larger font
plt.title('Strain from Model vs. Experiment', fontsize=16, weight='bold')
plt.ylabel('Strain Values (µε)', fontsize=14, fontweight='bold')
plt.xlabel('Time (sec)', fontsize=14, fontweight='bold')

# Show the plot
plt.tight_layout()


plt.show()


exit()













'''
       comparabledata = [
                        [
                        [idx1, idx2, idx3, idx4], [strainval1, strainval2, strainval3, strainval4]
                        ,...                                                                     ] 
                                                                                                 ]
                                                                                                 '''

datainquestion = zero_mean_each_dataset(comparabledata[300])


xtemp = datainquestion[0]
ytemp = datainquestion[1]

yplot = conv_adc_to_strain(ytemp)

datainquestion = [datainquestion[0], [x/100000 for x in yplot]]

# plt.scatter(xtemp, [x/1000 for x in yplot], color = 'black', label='ADC mesurements to strain')
# plt.show()
# exit()


# Unpack the xlist and ylist from the list
xlist, ylist = datainquestion

# Zip the two lists together to pair corresponding elements
zipped_lists = list(zip(xlist, ylist))

# Sort the zipped lists by the xlist (which is the first element in the pairs)
sorted_lists = sorted(zipped_lists, key=lambda pair: pair[0])

# Unzip the sorted list back into xlist and ylist
sorted_xlist, sorted_ylist = zip(*sorted_lists)

# Convert the result back to lists (since zip() returns tuples)
sorted_xlist = list(sorted_xlist)
sorted_ylist = list(sorted_ylist)

origonaldata = [sorted_xlist, sorted_ylist]


Xarr = []
Zarr = []
earr = []

for item in filtered_data:
    potentialx = item[0]
    potentialz = item[1]
    if potentialx not in Xarr:
        Xarr.append(potentialx)
    if potentialz not in Zarr:
        Zarr.append(potentialz)
    earr.append(item[2])


earr = np.array(earr)

# plt.imshow(earr.reshape([100,1]).T, cmap='turbo')
# plt.show()



# exit()
testedx = [37, 47, 48, 60]
newerr = earr.reshape([100,1])

scaledarr = newerr*10000000
dataset2 = []
# for row in scaledarr:
#     idx1 = np.mean(row[35:38])
#     idx2 = np.mean(row[45:47])
#     idx3 = np.mean(row[49:53])
#     idx4 = np.mean(row[58:62])
#     dataset2.append([idx1, idx2, idx3, idx4])

# dataset = [np.mean(earr[35:38]), np.mean(earr[45:47]), np.mean(earr[55:57]), np.mean(earr[58:62])]
dataset = [np.mean(earr[68:70]), np.mean(earr[57:59]), np.mean(earr[45:47]),np.mean(earr[30:32])  ]




corr_coeff = np.corrcoef(origonaldata[1], dataset)[0, 1]
print("Highest correlation coefficient:", corr_coeff)


# plt.scatter([1,2,3,4], dataset, color = 'blue', label='Baseline')
# plt.title("Baseline for a flat plate")
# plt.xlabel('Time (s)')
# plt.ylabel('Strain (uStrain)')
# plt.ylim([-.0000075, .0001])
# plt.show()


# # Find the dataset array with the highest correlation to the original
# best_match, highest_corr, best_index = find_highest_correlation(origonaldata[1], dataset)


# print("Best matching array:", best_match)
# print("Highest correlation coefficient:", highest_corr)
# print("Index of best match:", best_index)


plt.style.use('seaborn-darkgrid')
xdata = [-72, -22, 22, 72]
plt.scatter(xdata, dataset, color='g', label='Strain from Model', s=100, alpha=0.8, edgecolor='k')
plt.scatter(xdata, origonaldata[1], color='b', label='Strain from Experiment', s=100, alpha=0.8, edgecolor='k')


# plt.scatter([0,1,2,3], dataset, color='g', label='Strain from Model', s=100, alpha=0.8, edgecolor='k')
# plt.scatter(origonaldata[0], origonaldata[1], color='b', label='Strain from Experiment', s=100, alpha=0.8, edgecolor='k')
plt.xlim([-190, 190])
plt.ylim([-.0000075, .00005])

plt.legend(loc='upper right', fontsize=12)

# Title with larger font
plt.title('Strain from Model vs. Experiment', fontsize=16, weight='bold')
plt.ylabel('Strain Values (µε)', fontsize=14, fontweight='bold')
plt.xlabel('Position (mm)', fontsize=14, fontweight='bold')

# Show the plot
plt.tight_layout()


plt.show()