import serial
import time
import csv

# Configure serial ports
ser1 = serial.Serial('COM3', 9600)  # Replace 'COM3' with the first Arduino's port
ser2 = serial.Serial('COM4', 9600)  # Replace 'COM4' with the second Arduino's port

# Open the CSV file
with open('emg_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Timestamp', 'EMG1', 'EMG2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    while True:
        # Read data from both serial ports
        data1 = ser1.readline().decode().strip()
        data2 = ser2.readline().decode().strip()

        # Convert data to integers
        emg1 = int(data1)
        emg2 = int(data2)

        # Get current timestamp
        timestamp = time.time()

        # Write data to CSV
        writer.writerow({'Timestamp': timestamp, 'EMG1': emg1, 'EMG2': emg2})

        time.sleep(0.1)  # Adjust delay as needed