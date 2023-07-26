import serial
import time
import struct

class CoordinateData:
    def __init__(self):
        self.code = 0
        self.xpos = 0.0
        self.ypos = 0.0
        self.zpos = 0.0

def read_position_data(ser, data_size):
    start_time = time.time()
    while ser.in_waiting < data_size:
        if (time.time() - start_time) > 0.05:
            return None
    data = ser.read(data_size)
    return data

def main():
    # Open the serial port.
    ser = serial.Serial('/dev/ttyAMA0', 14400, timeout=1)
    ser.flush()

    while True:
        if ser.in_waiting > 0:
            coordinate_data = CoordinateData()
            code_data = ser.read(1)
            coordinate_data.code = int.from_bytes(code_data, byteorder='little')
            
            if coordinate_data.code == 1:
                pos_data = read_position_data(ser, 15)
                if pos_data:
                    xpos, ypos, zpos = struct.unpack('fff', pos_data)
                    coordinate_data.xpos = xpos
                    coordinate_data.ypos = ypos
                    coordinate_data.zpos = zpos

                    print(f"X: {coordinate_data.xpos}, Y: {coordinate_data.ypos}, Z: {coordinate_data.zpos}")
            else:
                while ser.in_waiting > 0:
                    incoming_byte = ser.read(1)
                    if incoming_byte == b'\x01':
                        pos_data = read_position_data(ser, 15)
                        if pos_data:
                            xpos, ypos, zpos = struct.unpack('fff', pos_data)
                            coordinate_data.xpos = xpos
                            coordinate_data.ypos = ypos
                            coordinate_data.zpos = zpos

                            print(f"X: {coordinate_data.xpos}, Y: {coordinate_data.ypos}, Z: {coordinate_data.zpos}")
                        break

if __name__ == "__main__":
    main()
