import serial
import matplotlib.pyplot as plt
from statistics import stdev, mean
from math import pi, cos, sin
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree  # Dùng KDTree để tìm láng giềng nhanh

class LidarData:
    """Lớp xử lý dữ liệu từ LiDAR và vẽ bản đồ khu vực với xóa điểm rời rạc."""

    # Hằng số lớp
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 500  # mm
    MAX_DATA_SIZE = 90  # Số điểm tối đa mỗi lần vẽ
    BAUDRATE = 115200
    NEIGHBOR_RADIUS = 50  # Bán kính tìm láng giềng (mm)
    MIN_NEIGHBORS = 4    # Số láng giềng tối thiểu để giữ điểm

    def __init__(self, port: str = 'COM4'):
        """Khởi tạo LiDAR với cổng serial và thiết lập đồ thị."""
        self.port = port
        self.ser: Optional[serial.Serial] = None
        self.data = {
            'angles': [],
            'distances': [],
            'speed': [],
            'x_coords': [],  # Tọa độ x (mm)
            'y_coords': [],  # Tọa độ y (mm)
        }

        # Thiết lập đồ thị 2D
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_ylim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_title("LiDAR Area Map (Noise Removed)")

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

    def _filter_data(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
        """Lọc dữ liệu để loại bỏ nhiễu dựa trên độ lệch chuẩn."""
        filtered_angles, filtered_distances = [], []
        window_size = 3

        for i in range(window_size, len(angles) - window_size):
            sample = distances[i - window_size:i + window_size + 1]
            if len(sample) < 2:
                continue
            sample_mean, sample_std = mean(sample), stdev(sample)
            if abs(distances[i] - sample_mean) < sample_std:
                filtered_angles.append(angles[i])
                filtered_distances.append(distances[i])

        return filtered_angles, filtered_distances

    def _to_cartesian(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
        """Chuyển đổi từ tọa độ phân cực sang Descartes."""
        x_coords = [dist * cos(angle) for angle, dist in zip(angles, distances)]
        y_coords = [dist * sin(angle) for angle, dist in zip(angles, distances)]
        return x_coords, y_coords

    def _remove_outliers(self, x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
        """Xóa các điểm rời rạc dựa trên số láng giềng trong bán kính."""
        if len(x_coords) < self.MIN_NEIGHBORS:
            return [], []  # Không đủ điểm để lọc

        # Tạo mảng tọa độ dạng NumPy
        points = np.array(list(zip(x_coords, y_coords)))
        tree = cKDTree(points)  # Dùng KDTree để tìm láng giềng nhanh

        # Tìm số láng giềng trong bán kính NEIGHBOR_RADIUS cho mỗi điểm
        neighbor_counts = tree.query_ball_point(points, r=self.NEIGHBOR_RADIUS, return_length=True)

        # Giữ lại các điểm có đủ láng giềng
        filtered_indices = [i for i, count in enumerate(neighbor_counts) if count >= self.MIN_NEIGHBORS]
        filtered_x = [x_coords[i] for i in filtered_indices]
        filtered_y = [y_coords[i] for i in filtered_indices]

        return filtered_x, filtered_y

    def _plot_map(self) -> None:
        """Vẽ bản đồ khu vực từ dữ liệu LiDAR sau khi xóa điểm rời rạc."""
        # Lọc dữ liệu ban đầu
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return

        # Chuyển sang tọa độ Descartes
        x_coords, y_coords = self._to_cartesian(angles, distances)

        # Xóa điểm rời rạc
        filtered_x, filtered_y = self._remove_outliers(x_coords, y_coords)

        # Cập nhật dữ liệu tổng thể
        self.data['x_coords'].extend(filtered_x)
        self.data['y_coords'].extend(filtered_y)

        # Xóa dữ liệu tạm thời
        self.data['angles'].clear()
        self.data['distances'].clear()

        # Vẽ bản đồ
        self.ax.clear()
        self.ax.set_xlim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_ylim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.grid(True)

        # Vẽ các điểm đã lọc
        if self.data['x_coords']:
            self.ax.scatter(self.data['x_coords'], self.data['y_coords'], c='blue', s=5, alpha=0.5)

        plt.draw()
        plt.pause(0.01)

    def update_data(self) -> None:
        """Cập nhật dữ liệu từ LiDAR và vẽ bản đồ."""
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
                        base_angle = int(sensor_data[0])
                        speed = int(sensor_data[1])
                        distances = [float(d) for d in sensor_data[2:6]]
                    except (ValueError, IndexError):
                        continue

                    for i, dist in enumerate(distances):
                        if self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE:
                            angle = (base_angle + i) * pi / 180
                            self.data['angles'].append(angle)
                            self.data['distances'].append(dist)
                            self.data['speed'].append(speed)

                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self._plot_map()

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

    def get_coordinates(self) -> Tuple[List[float], List[float]]:
        """Trả về tọa độ x, y."""
        return self.data['x_coords'], self.data['y_coords']


if __name__ == '__main__':
    try:
        sensor = LidarData(port='COM4')
        sensor.update_data()
    except Exception as e:
        print(f"Program failed: {e}")