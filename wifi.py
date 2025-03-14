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

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_ylim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_title("LiDAR Area Map (Noise Removed)")

        if not self._connect_wifi():
            raise ConnectionError(f"Failed to connect to {host}:{port}. Check IP/Port and try again.")

    def _connect_wifi(self) -> bool:
        try:
            print(f"Attempting to connect to {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # Tăng timeout lên 10 giây
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"WiFi connection error: {e}")
            return False

    # Các hàm còn lại giữ nguyên như code trước
    def _filter_data(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
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
        x_coords = [dist * cos(angle) for angle, dist in zip(angles, distances)]
        y_coords = [dist * sin(angle) for angle, dist in zip(angles, distances)]
        return x_coords, y_coords

    def _remove_outliers(self, x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
        if len(x_coords) < self.MIN_NEIGHBORS:
            return [], []
        points = np.array(list(zip(x_coords, y_coords)))
        tree = cKDTree(points)
        neighbor_counts = tree.query_ball_point(points, r=self.NEIGHBOR_RADIUS, return_length=True)
        filtered_indices = [i for i, count in enumerate(neighbor_counts) if count >= self.MIN_NEIGHBORS]
        filtered_x = [x_coords[i] for i in filtered_indices]
        filtered_y = [y_coords[i] for i in filtered_indices]
        return filtered_x, filtered_y

    def _plot_map(self) -> None:
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return
        x_coords, y_coords = self._to_cartesian(angles, distances)
        filtered_x, filtered_y = self._remove_outliers(x_coords, y_coords)
        self.data['x_coords'].extend(filtered_x)
        self.data['y_coords'].extend(filtered_y)
        self.data['angles'].clear()
        self.data['distances'].clear()
        self.ax.clear()
        self.ax.set_xlim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.set_ylim(-self.MAX_DISTANCE, self.MAX_DISTANCE)
        self.ax.grid(True)
        if self.data['x_coords']:
            self.ax.scatter(self.data['x_coords'], self.data['y_coords'], c='blue', s=5, alpha=0.5)
        plt.draw()
        plt.pause(0.01)

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