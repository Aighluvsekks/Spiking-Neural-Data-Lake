import serial
import csv
from datetime import datetime

# Configure your serial port and baud rate
# e.g., '/dev/ttyACM0' on Linux/Pi or 'COM3' on Windows
port = 'COM8'  # Update this to your actual port
ser = serial.Serial(port, 9600) 

with open("serial_sensor_log.csv", mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Sensor_1", "Sensor_2"])

    try:
        while True:
            if ser.in_waiting > 0:
                # Read the line, decode the bytes to a string, and strip the newline characters
                line = ser.readline().decode('utf-8').strip()
                
                # Split the incoming string "24.5,45.2" into a list ['24.5', '45.2']
                sensor_data = line.split(',')
                
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Combine the timestamp with the sensor data list
                row_to_write = [current_time] + sensor_data
                writer.writerow(row_to_write)
                
    except KeyboardInterrupt:
        print("Logging stopped.")
        ser.close()