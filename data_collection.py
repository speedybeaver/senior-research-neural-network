import serial
import csv
import numpy as np

def calculate_features(data):
    # Standard deviation
    std_dev = np.std(data)
    
    # Root mean square (RMS)
    rms = np.sqrt(np.mean(np.square(data)))
    
    # Minimum and maximum values
    min_value = np.min(data)
    max_value = np.max(data)
    
    # Zero crossings (number of times signal crosses zero)
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    
    # Average amplitude change (average of absolute differences between consecutive values)
    avg_amp_change = np.mean(np.abs(np.diff(data)))
    
    # Amplitude of first burst (max value within first quarter of data)
    amp_first_burst = np.max(data[:len(data) // 4])
    
    # Mean absolute value
    mean_abs_value = np.mean(np.abs(data))
    
    # Waveform length (sum of absolute differences between consecutive data points)
    wave_form_length = np.sum(np.abs(np.diff(data)))
    
    # Willison amplitude (number of times consecutive differences exceed a threshold)
    threshold = 0.1 * np.max(np.abs(data))  # Example threshold: 10% of max amplitude
    willison_amp = np.sum(np.abs(np.diff(data)) > threshold)
    
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
ser = serial.Serial('COM3', 9600)  # we change com3 to our port on the arduino

# Open CSV file (it'll create it if it doesn't exist)
with open('emg_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['std_dev', 'rms', 'min', 'max', 'zero_crossings', 'avg_amp_change', 'amp_first_burst', 'mean_abs_value', 'wave_form_length', 'willison_amp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    while True:
        data_buffer = []
        for _ in range(100):  # Adjust window size as needed
            data = ser.readline().decode().strip()
            data_buffer.append(int(data))
        
        # Calculate all features and write to CSV
        features = calculate_features(data_buffer)
        writer.writerow(features)