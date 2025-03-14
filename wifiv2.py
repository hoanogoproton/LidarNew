import socket
import matplotlib.pyplot as plt
from statistics import stdev, mean
from math import pi, cos, sin
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000
    MIN_DISTANCE = 500
    MAX_DATA_SIZE = 90
    NEIGHBOR_RADIUS = 50
    MIN_NEIGHBORS = 4

    def __init__(self, host: str = '192.168.0.106', port: int = 80):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.data = {
            'angles': [],
            'distances': [],
            'speed': [],
            'x_coords': [],
            'y_coords': [],
        }

        # Sử dụng chế độ tương tác của matplotlib
        plt.ion()  # Bật chế độ tương tác
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_ylim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_title("LiDAR Area Map (Noise Removed)")

        # Khởi tạo scatter plot một lần và tái sử dụng
        self.scatter = self.ax.scatter([], [], c='blue', s=5, alpha=0.5)

        if not self._connect_wifi():
            raise ConnectionError(f"Failed to connect to {host}:{port}. Check IP/Port and try again.")

    def _connect_wifi(self) -> bool:
        try:
            print(f"Attempting to connect to {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"WiFi connection error: {e}")
            return False

    def _filter_data(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
        # Sử dụng numpy để tăng tốc độ tính toán
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        window_size = 3
        filtered_indices = []

        # Vector hóa tính toán thay vì vòng lặp
        for i in range(window_size, len(distances_np) - window_size):
            sample = distances_np[i - window_size:i + window_size + 1]
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)
            if abs(distances_np[i] - sample_mean) < sample_std:
                filtered_indices.append(i)

        return angles_np[filtered_indices].tolist(), distances_np[filtered_indices].tolist()

    def _to_cartesian(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
        # Vector hóa tính toán tọa độ Descartes
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        x_coords = (distances_np * np.cos(angles_np)).tolist()
        y_coords = (distances_np * np.sin(angles_np)).tolist()
        return x_coords, y_coords

    def _remove_outliers(self, x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
        if len(x_coords) < self.MIN_NEIGHBORS:
            return [], []
        points = np.array(list(zip(x_coords, y_coords)))
        tree = cKDTree(points)
        neighbor_counts = tree.query_ball_point(points, r=self.NEIGHBOR_RADIUS, return_length=True)
        mask = neighbor_counts >= self.MIN_NEIGHBORS
        return points[mask, 0].tolist(), points[mask, 1].tolist()

    def _plot_map(self) -> None:
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return
        x_coords, y_coords = self._to_cartesian(angles, distances)
        filtered_x, filtered_y = self._remove_outliers(x_coords, y_coords)

        # Cập nhật dữ liệu thay vì xóa và vẽ lại
        self.data['x_coords'].extend(filtered_x)
        self.data['y_coords'].extend(filtered_y)
        self.data['angles'].clear()
        self.data['distances'].clear()

        # Cập nhật scatter plot thay vì vẽ mới
        self.scatter.set_offsets(np.c_[self.data['x_coords'], self.data['y_coords']])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_data(self) -> None:
        if not self.sock:
            print("No WiFi connection. Exiting...")
            return
        buffer = ""
        while True:
            try:
                data = self.sock.recv(1024).decode('utf-8', errors='ignore')
                if not data:
                    print("Connection closed by server.")
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    sensor_data = line.strip().split('\t')
                    if len(sensor_data) != self.DATA_LENGTH:
                        print(f"Invalid data length: {len(sensor_data)}")
                        continue
                    try:
                        base_angle = int(sensor_data[0])
                        speed = int(sensor_data[1])
                        distances = [float(d) for d in sensor_data[2:6]]
                    except (ValueError, IndexError) as e:
                        print(f"Data parse error: {e}")
                        continue
                    angles = [(base_angle + i) * pi / 180 for i in range(4)]
                    valid_mask = [(self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE) for dist in distances]
                    self.data['angles'].extend([a for a, v in zip(angles, valid_mask) if v])
                    self.data['distances'].extend([d for d, v in zip(distances, valid_mask) if v])
                    self.data['speed'].extend([speed] * sum(valid_mask))

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
        if self.sock:
            self.sock.close()
            print("WiFi connection closed.")
        plt.close(self.fig)
        print("Plot closed.")

    def get_coordinates(self) -> Tuple[List[float], List[float]]:
        return self.data['x_coords'], self.data['y_coords']


if __name__ == '__main__':
    try:
        sensor = LidarData(host='192.168.0.113', port=80)
        sensor.update_data()
    except Exception as e:
        print(f"Program failed: {e}")