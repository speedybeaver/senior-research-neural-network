import serial
import csv
import numpy as np
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import time
import matplotlib.pyplot as plt

# Parameters
SERIAL_PORT = 'COM3'    # Replace with the correct port
BAUD_RATE = 9600        # Ensure this matches your Arduino's baud rate
DURATION = 5            # Duration in seconds to collect data
CLASSIFICATION = 7      # Set the classification label for the current trial

# Function to calculate features for a single channel
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
    willison_amp = np.sum(np.abs(np.diff(data)) > 0.1)  # Threshold for amplitude change

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

# Configure serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Allow time for the serial connection to establish

# Open CSV file to save data
with open('emg_data.csv', 'a', newline='') as csvfile:
    
    data_buffer_ch1 = []  # Buffer for channel 1 data
    data_buffer_ch2 = []  # Buffer for channel 2 data
    start_time = time.time()

    try:
        while time.time() - start_time < DURATION:
            # Read data from serial if available
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                try:
                    # Split the line by comma to get channel 1 and channel 2 values
                    ch1_value, ch2_value = map(float, line.split(','))

                    print(f"{ch1_value}, {ch2_value}");

                    data_buffer_ch1.append(ch1_value)
                    data_buffer_ch2.append(ch2_value)
                    #print("Current buffer length:", len(data_buffer_ch1), len(data_buffer_ch2))  # Debug: check buffer length
                except ValueError:
                    # Skip lines that cannot be converted to float or do not have two values
                    continue
    except KeyboardInterrupt:
        print("Data collection stopped manually.")
    finally:
        # Center data buffers around their means
        data_buffer_ch1 = data_buffer_ch1 - np.mean(data_buffer_ch1)
        data_buffer_ch2 = data_buffer_ch2 - np.mean(data_buffer_ch2)

        b, a = iirnotch(60, 30, fs=1000)
        emg_data_notch_ch1 = lfilter(b, a, data_buffer_ch1)
        emg_data_notch_ch2 = lfilter(b, a, data_buffer_ch2)

        # Low-pass filter to reduce high-frequency noise
        cutoff = 50  # Cutoff frequency
        order = 4
        b, a = butter(order, cutoff, fs=1000, btype='lowpass')
        emg_data_filtered_ch1 = lfilter(b, a, emg_data_notch_ch1)
        emg_data_filtered_ch2 = lfilter(b, a, emg_data_notch_ch2)

        # Calculate features for each channel independently
        features_ch1 = calculate_features(emg_data_filtered_ch1)
        features_ch2 = calculate_features(emg_data_filtered_ch2)


        # starts adding to the csv

        fieldnames = [
                'std_dev_ch1', 'std_dev_ch2', 
                'rms_ch1', 'rms_ch2', 
                'min_ch1', 'min_ch2', 
                'max_ch1', 'max_ch2', 
                'zero_crossings_ch1', 'zero_crossings_ch2', 
                'avg_amp_change_ch1', 'avg_amp_change_ch2', 
                'amp_first_burst_ch1', 'amp_first_burst_ch2', 
                'mean_abs_value_ch1', 'mean_abs_value_ch2', 
                'wave_form_length_ch1', 'wave_form_length_ch2', 
                'willison_amp_ch1', 'willison_amp_ch2',
                'classification',
                *['ch1_data_' + str(i) for i in range(len(data_buffer_ch1))],
                *['ch2_data_' + str(i) for i in range(len(data_buffer_ch2))]
            ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer.writeheader()

        row = {
            **{f"{k}_ch1": v for k, v in features_ch1.items()},
            **{f"{k}_ch2": v for k, v in features_ch2.items()},
            'classification': CLASSIFICATION,
            # Add raw data points to the row dictionary
            **dict((f'ch1_data_{i}', data_buffer_ch1[i]) for i in range(len(data_buffer_ch1))),
            **dict((f'ch2_data_{i}', data_buffer_ch2[i]) for i in range(len(data_buffer_ch2)))
        }

        # Write the row to the CSV file
        writer.writerow(row)

        # Clear buffers and close serial connection
        data_buffer_ch1 = []
        data_buffer_ch2 = []
        ser.close()

print("Data collection complete. CSV file created.")
