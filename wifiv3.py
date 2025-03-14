import socket
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Optional, Tuple
from math import pi

class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 300   # mm
    MAX_DATA_SIZE = 90
    NEIGHBOR_RADIUS = 50
    MIN_NEIGHBORS = 4
    GRID_SIZE = 50      # Kích thước mỗi ô lưới (mm)

    def __init__(self, host: str = '192.168.0.113', port: int = 80):
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
        self.grid = None  # Lưu trữ occupancy grid

        # Thiết lập matplotlib
        plt.ion()  # Chế độ tương tác
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("LiDAR Occupancy Grid Map")
        self.ax.grid(True)

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
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        window_size = 3
        filtered_indices = []
        for i in range(window_size, len(distances_np) - window_size):
            sample = distances_np[i - window_size:i + window_size + 1]
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)
            if abs(distances_np[i] - sample_mean) < sample_std:
                filtered_indices.append(i)
        return angles_np[filtered_indices].tolist(), distances_np[filtered_indices].tolist()

    def _to_cartesian(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
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

    def create_occupancy_grid(self) -> np.ndarray:
        """Tạo occupancy grid từ tọa độ x, y"""
        # Kích thước lưới dựa trên MAX_DISTANCE (mm), chia thành các ô GRID_SIZE (mm)
        grid_size = int(2 * self.MAX_DISTANCE / self.GRID_SIZE)
        if self.grid is None:
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        for x, y in zip(self.data['x_coords'], self.data['y_coords']):
            # Chuyển tọa độ thực tế sang chỉ số lưới
            grid_x = int((x + self.MAX_DISTANCE) / self.GRID_SIZE)
            grid_y = int((y + self.MAX_DISTANCE) / self.GRID_SIZE)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                self.grid[grid_x, grid_y] = 255  # Đánh dấu ô bị chiếm

        return self.grid

    def _plot_map(self) -> None:
        """Cập nhật và hiển thị occupancy grid"""
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return
        x_coords, y_coords = self._to_cartesian(angles, distances)
        filtered_x, filtered_y = self._remove_outliers(x_coords, y_coords)

        # Thêm dữ liệu mới vào danh sách tọa độ
        self.data['x_coords'].extend(filtered_x)
        self.data['y_coords'].extend(filtered_y)
        self.data['angles'].clear()
        self.data['distances'].clear()

        # Cập nhật occupancy grid
        grid = self.create_occupancy_grid()

        # Hiển thị lưới
        self.ax.clear()
        self.ax.imshow(grid, cmap='gray', origin='lower', extent=(-self.MAX_DISTANCE, self.MAX_DISTANCE, -self.MAX_DISTANCE, self.MAX_DISTANCE))
        self.ax.set_title("LiDAR Occupancy Grid Map")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
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