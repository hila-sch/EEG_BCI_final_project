import serial
import time

aduinoData = serial.Serial('com10', 115200)
time.sleep(1)

while True:
    cmd = input("Enter command: ")
    cmd = cmd + "\r"
    aduinoData.write(cmd.encode())
