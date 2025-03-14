import serial
import matplotlib.pyplot as plt
from statistics import stdev, mean
from math import pi
from typing import List, Optional


class LidarData:
    """Lớp xử lý dữ liệu từ LiDAR và hiển thị trên biểu đồ phân cực."""

    # Hằng số lớp
    DATA_LENGTH = 7  # Số trường trong mỗi dòng dữ liệu từ LiDAR
    MAX_DISTANCE = 300  # Khoảng cách tối đa (mm)
    MIN_DISTANCE = 50  # Khoảng cách tối thiểu (mm)
    MAX_DATA_SIZE = 90  # Số điểm tối đa trước khi vẽ (1 điểm mỗi 4 độ)
    BAUDRATE = 115200  # Tốc độ baud cho kết nối serial

    def __init__(self, port: str = 'COM4'):
        """Khởi tạo LiDAR với cổng serial và thiết lập đồ thị."""
        self.port = port
        self.ser: Optional[serial.Serial] = None
        self.data = {
            'angles': [],  # Góc (radian)
            'distances': [],  # Khoảng cách (mm)
            'speed': [],  # Tốc độ quay (RPM)
        }

        # Thiết lập biểu đồ phân cực
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax.set_rmax(self.MAX_DISTANCE)  # Đặt giới hạn bán kính khớp với MAX_DISTANCE

        # Kết nối serial
        if not self._connect_serial():
            raise ConnectionError(f"Failed to connect to {port}. Check port and try again.")

    def _connect_serial(self) -> bool:
        """Kết nối với cổng serial."""
        try:
            self.ser = serial.Serial(self.port, self.BAUDRATE, timeout=1)
            self.ser.reset_input_buffer()
            print(f"Connected to {self.port} @ {self.BAUDRATE} baud")
            return True
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            return False

    def _filter_data(self, angles: List[float], distances: List[float]) -> tuple[List[float], List[float]]:
        """Lọc dữ liệu để loại bỏ nhiễu dựa trên độ lệch chuẩn."""
        filtered_angles, filtered_distances = [], []
        window_size = 3  # Kích thước cửa sổ lọc (2 * window_size + 1)

        for i in range(window_size, len(angles) - window_size):
            sample = distances[i - window_size:i + window_size + 1]
            if len(sample) < 2:  # Cần ít nhất 2 điểm để tính stdev
                continue
            sample_mean, sample_std = mean(sample), stdev(sample)
            if abs(distances[i] - sample_mean) < sample_std:
                filtered_angles.append(angles[i])
                filtered_distances.append(distances[i])

        return filtered_angles, filtered_distances

    def _plot_data(self) -> None:
        """Vẽ dữ liệu lên biểu đồ phân cực."""
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        self.ax.clear()
        self.ax.plot(angles, distances, '.', markersize=5)  # Tăng kích thước điểm cho dễ nhìn
        self.ax.set_rmax(self.MAX_DISTANCE)
        self.data['angles'].clear()
        self.data['distances'].clear()
        plt.draw()
        plt.pause(0.01)  # Thời gian dừng để cập nhật giao diện

    def update_data(self) -> None:
        """Cập nhật dữ liệu từ LiDAR và vẽ khi đủ điểm."""
        if not self.ser:
            print("No serial connection. Exiting...")
            return

        while True:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    sensor_data = line.split('\t')

                    if len(sensor_data) != self.DATA_LENGTH:
                        continue

                    try:
                        base_angle = int(sensor_data[0])  # Góc khởi đầu
                        speed = int(sensor_data[1])  # Tốc độ (RPM)
                        distances = [float(d) for d in sensor_data[2:6]]  # 4 khoảng cách
                    except (ValueError, IndexError):
                        continue

                    # Xử lý từng khoảng cách
                    for i, dist in enumerate(distances):
                        if self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE:
                            angle = (base_angle + i) * pi / 180  # Chuyển sang radian
                            self.data['angles'].append(angle)
                            self.data['distances'].append(dist)
                            self.data['speed'].append(speed)
                            # Debug (giảm tần suất in)
                            if len(self.data['angles']) % 10 == 0:
                                print(f"Speed: {speed} RPM, Angle: {base_angle + i}°, Dist: {dist:.0f}mm")

                    # Vẽ khi đủ dữ liệu
                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self._plot_data()

            except KeyboardInterrupt:
                print("Stopped by user.")
                self._cleanup()
                break
            except Exception as e:
                print(f"Error reading data: {e}")
                continue

    def _cleanup(self) -> None:
        """Dọn dẹp tài nguyên khi thoát."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")
        plt.close(self.fig)
        print("Plot closed.")

    def get_distances(self) -> List[float]:
        """Trả về danh sách khoảng cách."""
        return self.data['distances']

    def get_angles(self) -> List[float]:
        """Trả về danh sách góc."""
        return self.data['angles']


if __name__ == '__main__':
    try:
        sensor = LidarData(port='COM4')
        sensor.update_data()
    except Exception as e:
        print(f"Program failed: {e}")