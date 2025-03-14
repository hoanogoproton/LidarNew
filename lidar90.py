import serial
import matplotlib.pyplot as plt
import statistics
import math


class LidarData():

    def __init__(self):
        self.DATA_LENGTH = 7  # Độ dài gói dữ liệu
        self.MAX_DISTANCE = 300  # Giới hạn khoảng cách tối đa (mm)
        self.MIN_DISTANCE = 50  # Giới hạn khoảng cách tối thiểu (mm)
        self.port = 'COM4'  # Cổng COM
        self.MAX_DATA_SIZE = 90  # Giảm từ 360 xuống 90 (1 điểm mỗi 4 độ)
        self.ser = None
        self.BAUDRATE = 115200

        self.data = {
            'angles': [],
            'distances': [],
            'speed': [],
            'signal_strength': [],
            'checksum': []
        }

        # Thiết lập đồ thị
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax.set_rmax(3000)  # Sửa từ 300 thành 3000 để khớp MAX_DISTANCE

        if not self.connectSerial(self.port, self.BAUDRATE):
            print("Failed to connect to serial port. Please check the port and try again.")
            exit(1)

    def connectSerial(self, port: str, baudrate: int) -> bool:
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.ser.reset_input_buffer()
            print(f'Serial connection established @ {port}')
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def plotData(self) -> None:
        angles, distances = [], []

        # Lọc dữ liệu (giữ nguyên logic nhưng xử lý ít điểm hơn do MAX_DATA_SIZE giảm)
        for p in range(3, len(self.data['angles']) - 3):
            if p > len(self.data['angles']) - 3:
                break
            sample = self.data['distances'][p - 3:p + 3]
            std = statistics.stdev(sample)
            if abs(self.data['distances'][p] - statistics.mean(sample)) < std:
                angles.append(self.data['angles'][p])
                distances.append(self.data['distances'][p])

        self.ax.clear()
        plt.plot(angles, distances, ".")
        self.ax.set_rmax(self.MAX_DISTANCE)
        self.data['angles'].clear()
        self.data['distances'].clear()
        plt.draw()
        plt.pause(0.01)  # Tăng từ 0.001 lên 0.01 để giảm tần suất cập nhật

    def updateData(self) -> None:
        if self.ser is None:
            print("Serial connection not established. Exiting...")
            return
        while True:
            try:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode().rstrip()
                        sensorData = line.split('\t')
                    except:
                        continue

                    if len(sensorData) == self.DATA_LENGTH:
                        for i in range(2, 6):
                            try:
                                angle = (int(sensorData[0]) + i - 1) * math.pi / 180
                                dist = float(sensorData[i])
                                print(
                                    f'speed: {int(sensorData[1])} RPM, angle: {round(angle * 180 / math.pi)}, dist: {round(dist)}')
                            except:
                                continue

                            if dist >= self.MIN_DISTANCE and dist <= self.MAX_DISTANCE:
                                self.data['angles'].append(angle)
                                self.data['distances'].append(dist)
                                self.data['checksum'].append(sensorData[-1])
                                self.data['speed'].append(sensorData[1])

                            if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                                self.plotData()

            except KeyboardInterrupt:
                exit()

    def getDistances(self) -> list:
        return self.data['distances']

    def getAngles(self) -> list:
        return self.data['angles']


if __name__ == '__main__':
    sensor = LidarData()
    sensor.updateData()