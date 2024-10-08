## Developing for colored heat matp
import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import griddata
import time
import threading
from PIL import Image, ImageTk  # Pillow library for more image formats
import sv_ttk
from matplotlib.colors import Normalize
import random


global selected_value, com_ports, selected_values2, threads, com_data, grid1, grid2

# Initialize the 
ser = [None]  # Serial object (initially None)
# Data storage: Pandas DataFrame to hold timestamp  ed strain values for each location
data = pd.DataFrame(columns=["Timestamp", "ID", "Strain", "Temp"])
com_ports = []
cBoxes = []

def screen1():
    # Clear the previous screen (if any)
    global selected_value
    selected_value = tk.IntVar()
    # This is where the magic happens
    # sv_ttk.set_theme("dark")
    root.geometry("600x400") 

    for widget in root.winfo_children():
        widget.destroy()
    ## need to reset global variables like selected value   

    # Create a frame for buttons at the top
    button_frame = ttk.Frame(root)
    button_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    # Add drop down
    drop_down_1 = ttk.Label(button_frame, text="Number of Sensors: ")
    drop_down_1.pack(side=tk.LEFT, padx=5)
    no_sensors = ttk.Combobox(button_frame, values=[1,2,3,4,5,6], width=3)
    no_sensors.pack(side=tk.LEFT, padx=5)

    # Add buttons to the top
    btn1 = ttk.Button(button_frame, text="Next", command=screen2)
    btn1.pack(side=tk.LEFT, padx=5)

    btn2 = ttk.Button(button_frame, text="Quit", command=root.destroy)
    btn2.pack(side=tk.LEFT, padx=5)

    btn3 = ttk.Button(button_frame, text="Reset", command=screen1)
    btn3.pack(side=tk.LEFT, padx=5)

    no_sensors.bind("<<ComboboxSelected>>", lambda event: selected_value.set(no_sensors.get()))




def screen2():
    # Clear the previous screen (if any)
    global selected_value, cBoxes
    for widget in root.winfo_children():
        widget.destroy()
    


    # Create a frame for buttons at the top
    button_frame = ttk.Frame(root)
    button_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    label_1 = ttk.Label(button_frame, text=f"Selected value: {selected_value.get()}")
    label_1.pack(side=tk.LEFT, padx=5)

    # Add buttons to the top
    btn1 = ttk.Button(button_frame, text="Next", command=switch_to_screen_3)
    btn1.pack(side=tk.LEFT, padx=5)

    btn2 = ttk.Button(button_frame, text="Quit", command=root.destroy)
    btn2.pack(side=tk.LEFT, padx=5)

    btn3 = ttk.Button(button_frame, text="Reset", command=screen1)
    btn3.pack(side=tk.LEFT, padx=5)

    # retrieve a list of all available ports
    ports = list_ports()

    drop_down_frame = ttk.Frame(root)
    drop_down_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=10)


    # Create a list of variables dynamically
    cBoxes = [None] * int(selected_value.get())  # This creates a list with the number of com ports


    for idx, i in enumerate(cBoxes):
        ttk.Label(drop_down_frame, text='Select Sensor ' + str(idx+1) + ' COM Port: ').pack(side='top', padx=5)
        cBoxes[idx] = ttk.Combobox(drop_down_frame, values=ports, width=6)
        cBoxes[idx].pack(side='top', padx=5)
    



def screen3():
    # Clear the previous screen (if any)
    global selected_value, canvas1, canvas2, com_ports, cBoxes, threads
    selected_value = tk.IntVar()

    for widget in root.winfo_children():
        widget.destroy()

    # Create a frame for buttons at the top
    button_frame = ttk.Frame(root)
    button_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    btn2 = ttk.Button(button_frame, text="Quit", command=root.destroy)
    btn2.pack(side=tk.LEFT, padx=5)

    btn3 = ttk.Button(button_frame, text="Reset", command=screen1)
    btn3.pack(side=tk.LEFT, padx=5)

    # Start/Stop buttons
    start_button = ttk.Button(button_frame, text="Start Logging", command=start_logging)
    start_button.pack(side=tk.LEFT, padx=10, pady=10)

    stop_button = ttk.Button(button_frame, text="Stop Logging", command=stop_logging)#, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a canvas to display the first scatter plot
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # Create a canvas to display the second scatter plot
    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=10, pady=10)



# Function to update the plot
def update_plot(current_time):
    global canvas1, canvas2, com_data, data_lock, grid1, grid2
    with data_lock:    
        ax1.clear()
        ax2.clear()
        data = pd.concat(com_data.values(), ignore_index=True)

        timestamp_data = data[data["Timestamp"] <= current_time]
        # print(timestamp_data)

        

        for loc_id, (x, y) in locations.items():
            strain_values = timestamp_data[timestamp_data["ID"] == loc_id]
            x_loc = x
            y_loc = y
            temp_values = timestamp_data[timestamp_data["ID"] == loc_id]

            

            if not strain_values.empty:
                latest_strain_value = strain_values.iloc[-1]  # Get the latest strain value for this location
                grid1[(x_loc, y_loc)] = latest_strain_value["Strain"]
                ax1.imshow(grid1, interpolation='gaussian', cmap='turbo', vmin = 0, vmax = 1024)

            if not temp_values.empty:
                latest_temp_value = temp_values.iloc[-1]  # Get the latest strain value for this location
                grid2[(x_loc, y_loc)] = latest_temp_value["Temp"]
                ax2.imshow(grid2, interpolation='gaussian', cmap='turbo',  vmin = 10, vmax = 45)



                # ax1.set_title('haha')
                # latest_strain_value = strain_values.iloc[-1]  # Get the latest strain value for this location
                # ax2.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', 
                # cmap='hot', aspect='auto', norm=Normalize(vmin=min(z), vmax=max(z)))


            # if not strain_values.empty:
                # latest_strain_value = strain_values.iloc[-1]  # Get the latest strain value for this location
                # ax1.scatter(x, y, s=200, c=latest_strain_value["Strain"], cmap='Reds', vmin=0, vmax=100)
                # ax1.text(x, y, f"ID: {loc_id}\nStrain: {latest_strain_value['Strain']:.2f}", 
                #         ha="center", va="center", fontsize=10)
            # if not temp_values.empty:
            #     latest_temp_value = temp_values.iloc[-1]  # Get the latest strain value for this location
            #     ax2.scatter(x, y, s=200, c=latest_temp_value["Temp"], cmap='Reds', vmin=0, vmax=100)
            #     ax2.text(x, y, f"ID: {loc_id}\nTemp: {latest_temp_value['Temp']:.2f}", 
            #             ha="center", va="center", fontsize=10)
            

        ax1.set_xlim(0, 3)
        ax1.set_ylim(-.5, .5)
        ax1.set_title("Strain Map")# - Time: {current_time:.2f}",fontsize=20)

        # ax1.set_title(f"Strain Map - Time: {current_time:.2f}",fontsize=20)
        canvas1.draw()

        ax2.set_xlim(0, 3)
        ax2.set_ylim(-.5, .5)
        # ax2.set_title(f"Temp Map - Time: {current_time:.2f}",fontsize=20)

        ax2.set_title("Temperature Map")# - Time: {current_time:.2f}",fontsize=20)
        canvas2.draw()

# Function to read serial data and update the data storage
def read_serial(com_port):
    global com_data, data_lock, stop_event
    ser = serial.Serial(com_port, baudrate=115200, timeout=1)  # Adjust baudrate and timeout as needed
    while not stop_event.is_set():
        if ser and ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            try:
                # Assuming the data is sent in format: ID,Strain,Temp
                id_, strain, temp = map(float, line.split(","))
                timestamp = time.time()
                new_row = {"Timestamp": timestamp, "ID": id_, "Strain": strain, "Temp": temp}
                new_data = pd.DataFrame([new_row])
                with data_lock:
                    # If COM port data doesn't exist yet, create a new DataFrame
                    if ser not in com_data:
                        com_data[ser] = new_data
                    else:
                        # Append new data to the existing DataFrame for that COM port
                        com_data[ser] = pd.concat([com_data[ser], new_data], ignore_index=True)
                update_plot(timestamp)

            except ValueError:
                print("Error parsing the data.")
            time.sleep(.5)

# Function to list available COM ports
def list_ports():
    ports = list(serial.tools.list_ports.comports())
    return [port.device for port in ports]

# Function to start logging data
def start_logging():
    global ser, logging, selected_values2, threads, com_data, data_lock, stop_event
    ser = [None] * int(len(selected_values2)) 
    stop_event.clear()
    for com_port in selected_values2:
        thread = threading.Thread(target=read_serial, args=(com_port, ))
        threads.append(thread)

    for thread in threads:
        thread.start()

    logging = True   
    start_event.set()

# Function to stop logging data
def stop_logging():
    global logging, threads
    logging = False
    stop_event.set()
    # for thread in threads:
    #     if ser:
    #         ser.close()
    # start_button.config(state=tk.NORMAL)
    # stop_button.config(state=tk.DISABLED)
root = tk.Tk()
root.title("SENSE-VT")

def switch_to_screen_3():
    global selected_values2, cBoxes
    selected_values2.clear()  # Clear previous values
    for cBox in cBoxes:
        value = cBox.get()
        selected_values2.append(value)
    screen3()

# Configure the root grid layout
root.columnconfigure(0, weight=1)  # Make column 0 stretchable
root.columnconfigure(1, weight=1)  # Make column 0 stretchable
root.rowconfigure(1, weight=1)  # Make row 1 stretchable for the first plot
root.rowconfigure(2, weight=1)  # Make row 2 stretchable for the second plot

selected_value = tk.IntVar()
selected_value = tk.StringVar()

# Create two scatter plots with matplotlib
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


# Plot the data on the two axes
# ax1.scatter(x, y1, color='blue')
ax1.set_title('Strain Plot')

# ax2.scatter(x, y2, color='red')
ax2.set_title('Temp Plot')


# Placeholder for location mapping (adjust this to your actual locations)
locations = {2: (0, 0), 1: (0, 1), 3: (0, 2), 4: (0, 3)}




# declare the global variable for soring the ports you'd like to look at
selected_values2 = []
threads = []
com_data = {}
grid1 = np.zeros([1,4])
grid2 = np.zeros([1,4])



plt.ion()
start_event = threading.Event()
stop_event = threading.Event()
data_lock = threading.Lock()  # Lock to ensure thread-safe data updates

# Start with the first screen
screen1()


logging = False
root.mainloop()
