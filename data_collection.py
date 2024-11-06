import serial
import csv
import numpy as np
import time

# Parameters
SERIAL_PORT = 'COM3'   # Replace with the correct port
BAUD_RATE = 9600       # Match with your Arduino code
DURATION = 5          # Duration in seconds to collect data

# Function to calculate features
def calculate_features(data):
    std_dev = np.std(data)
    rms = np.sqrt(np.mean(np.square(data)))
    min_value = np.min(data)
    max_value = np.max(data)
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    avg_amp_change = np.mean(np.abs(np.diff(data)))
    amp_first_burst = data[0]
    mean_abs_value = np.mean(np.abs(data))
    wave_form_length = np.sum(np.abs(np.diff(data)))
    willison_amp = np.sum(np.abs(np.diff(data)) > 0.1)  # Threshold of 0.1

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
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Give time for the serial connection to establish

# Open CSV file for writing
with open('emg_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['std_dev', 'rms', 'min', 'max', 'zero_crossings', 
                  'avg_amp_change', 'amp_first_burst', 'mean_abs_value', 
                  'wave_form_length', 'willison_amp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    data_buffer = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            # Read data from serial
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                try:
                    value = float(line)  # Convert the reading to an integer
                    data_buffer.append(value)
                    data_buffer.append(value)
                    print("Current buffer length:", len(data_buffer))  # Debug: check buffer length
                except ValueError:
                    # Skip lines that cannot be converted to an integer
                    continue
    except KeyboardInterrupt:
        print("Data collection stopped manually.")
    finally:
        features = calculate_features(data_buffer)
        mean = np.mean(features)
        features = features - mean # Centers data around mean
        print("Features calculated:", features)  # Debug: Confirm feature calculation
        writer.writerow(features)  # Write features to CSV
        data_buffer = []  # Clear buffer after processing
        ser.close()  # Ensure the serial connection is closed

print("Data collection complete. CSV file created.")
