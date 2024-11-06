import serial
import csv
import numpy as np

def calculate_features(data):
    # Split data into 8 electrodes, each with 10 values (features)
    electrode_data = np.reshape(data, (8, 10))
    
    # Extract each feature across the electrodes
    std_dev = electrode_data[:, 0].mean()
    rms = electrode_data[:, 1].mean()
    min_value = electrode_data[:, 2].min()
    max_value = electrode_data[:, 3].max()
    zero_crossings = electrode_data[:, 4].sum()
    avg_amp_change = electrode_data[:, 5].mean()
    amp_first_burst = electrode_data[:, 6].max()
    mean_abs_value = electrode_data[:, 7].mean()
    wave_form_length = electrode_data[:, 8].sum()
    willison_amp = electrode_data[:, 9].sum()
    
    return {
        'std_dev': std_dev,
        'rms': rms,
        'min': min_value,
        'max': max_value,
        'zero_crossings': zero_crossings,
        'avg_amp_change': avg_amp_change,
        'amp_first_burst': amp_first_burst,
        'mean_abs_value': mean_abs_value,
        'wave_form_length': wave_form_length,
        'willison_amp': willison_amp
    }

# Configure serial port
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino's port

# Open CSV file
with open('emg_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['std_dev', 'rms', 'min', 'max', 'zero_crossings', 'avg_amp_change', 'amp_first_burst', 'mean_abs_value', 'wave_form_length', 'willison_amp', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    while True:
        data_buffer = []
        for _ in range(80):  # Read 80 data points to match the structure of 8 electrodes with 10 features
            data = ser.readline().decode().strip()
            data_buffer.append(float(data))
        
        # Calculate features from the data
        features = calculate_features(data_buffer)
        
        # Assign a label if needed; replace '0' with actual labels if they are available
        features['label'] = 0
        
        # Write to CSV
        writer.writerow(features)
