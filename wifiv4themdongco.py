import socket
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Optional, Tuple
from math import pi
import tkinter as tk
import threading
import queue
import time


class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 300  # mm
    MAX_DATA_SIZE = 200  # Tăng từ 90 lên 200 để tích lũy nhiều điểm hơn trước khi vẽ
    NEIGHBOR_RADIUS = 48
    MIN_NEIGHBORS = 3
    GRID_SIZE = 50  # Kích thước mỗi ô lưới (mm)

    def __init__(self, host: str = '192.168.0.120', port: int = 80):
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
        self.grid = None
        self.robot_distance = 0.0
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Thêm các thông số để tính toán khoảng cách
        self.wheel_diameter = 7.0  # cm, giống với Arduino
        self.ppr = 500  # pulses per revolution, giống với Arduino
        self.pi = 3.1416
        self.wheel_circumference = self.wheel_diameter * self.pi

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("LiDAR Occupancy Grid Map with Robot Position")
        self.ax.grid(True)

        if not self._connect_wifi():
            raise ConnectionError(f"Failed to connect to {host}:{port}. Check IP/Port and try again.")

        self.command_thread = threading.Thread(target=self._handle_commands, daemon=True)
        self.command_thread.start()
        self.process_thread = threading.Thread(target=self.process_data, daemon=True)
        self.process_thread.start()
        self.plot_queue = queue.Queue()  # Hàng đợi để gửi tín hiệu cập nhật đồ họa

    def _connect_wifi(self) -> bool:
        try:
            print(f"Attempting to connect to {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            print(f"Connected to {self.host}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"WiFi connection error: {e}")
            return False

    def _filter_data(self, angles: List[float], distances: List[float]) -> Tuple[List[float], List[float]]:
        filtered_angles = []
        filtered_distances = []
        for a, d in zip(angles, distances):
            if self.MIN_DISTANCE <= d <= self.MAX_DISTANCE:
                filtered_angles.append(a)
                filtered_distances.append(d)
        return filtered_angles, filtered_distances

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
        grid_size = int(2 * self.MAX_DISTANCE / self.GRID_SIZE)
        if self.grid is None:
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for x, y in zip(self.data['x_coords'], self.data['y_coords']):
            grid_x = int((x + self.MAX_DISTANCE) / self.GRID_SIZE)
            grid_y = int((y + self.MAX_DISTANCE) / self.GRID_SIZE)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                self.grid[grid_x, grid_y] = 255
        return self.grid

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
        grid = self.create_occupancy_grid()

        if not hasattr(self, 'background'):
            self.ax.imshow(grid, cmap='gray', origin='lower',
                           extent=(-self.MAX_DISTANCE, self.MAX_DISTANCE, -self.MAX_DISTANCE, self.MAX_DISTANCE))
            self.ax.plot(0, 0, 'ro', markersize=10, label='Robot Position')
            self.ax.set_title(f"LiDAR Occupancy Grid Map - Distance: {self.robot_distance:.2f} cm")
            self.ax.legend()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            self.ax.imshow(grid, cmap='gray', origin='lower',
                           extent=(-self.MAX_DISTANCE, self.MAX_DISTANCE, -self.MAX_DISTANCE, self.MAX_DISTANCE))
            self.ax.plot(0, 0, 'ro', markersize=10)
            self.ax.set_title(f"LiDAR Occupancy Grid Map - Distance: {self.robot_distance:.2f} cm")
            self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def process_data(self):
        while True:
            if not self.data_queue.empty():
                # Gửi tín hiệu rằng có dữ liệu mới cần vẽ
                self.plot_queue.put(True)
            time.sleep(0.01)  # Giảm tải CPU

    def update_data(self) -> None:
        buffer = ""
        while True:
            try:
                data = self.sock.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    print("Connection closed by server.")
                    break
                buffer += data
            except BlockingIOError:
                time.sleep(0.01)
                continue

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                # In dữ liệu thô nhận được
                print(f"Raw data received: {line}")
                sensor_data = line.strip().split('\t')
                if len(sensor_data) != 8:  # Đảm bảo dữ liệu có 8 phần tử
                    continue
                try:
                    base_angle = int(sensor_data[0])
                    speed = int(sensor_data[1])
                    distances = [float(d) for d in sensor_data[2:6]]
                    encoder_count = int(sensor_data[7].strip())  # Nhận encoder_count thay vì distance
                    # Tính toán khoảng cách từ encoder_count
                    self.robot_distance = (encoder_count * self.wheel_circumference) / self.ppr
                    angles = [(base_angle + i) * pi / 180 for i in range(4)]
                    valid_mask = [(self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE) for dist in distances]
                    self.data['angles'].extend([a for a, v in zip(angles, valid_mask) if v])
                    self.data['distances'].extend([d for d, v in zip(distances, valid_mask) if v])
                    self.data['speed'].extend([speed] * sum(valid_mask))
                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self.data_queue.put(True)
                    if len(self.data['angles']) > 500:
                        indices = np.random.choice(len(self.data['angles']), 200, replace=False)
                        self.data['angles'] = [self.data['angles'][i] for i in indices]
                        self.data['distances'] = [self.data['distances'][i] for i in indices]
                        self.data['speed'] = [self.data['speed'][i] for i in indices]
                except (ValueError, IndexError):
                    continue

    def send_command(self, command: str) -> None:
        self.command_queue.put(command)

    def _handle_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get()
                if self.sock:
                    self.sock.sendall((command + '\n').encode('utf-8'))
                    print(f"Sent command: {command}")
                self.command_queue.task_done()
            except Exception as e:
                print(f"Failed to send command: {e}")
                time.sleep(0.01)

    def _cleanup(self) -> None:
        if self.sock:
            self.sock.close()
            print("WiFi connection closed.")
        plt.close(self.fig)
        print("Plot closed.")

    def get_coordinates(self) -> Tuple[List[float], List[float]]:
        return self.data['x_coords'], self.data['y_coords']


class ControlGUI:
    def __init__(self, lidar_data):
        self.lidar_data = lidar_data
        self.root = tk.Tk()
        self.root.title("Điều Khiển Động Cơ")

        # Nút Tiến
        self.forward_button = tk.Button(self.root, text="Tiến")
        self.forward_button.pack(pady=10)
        self.forward_button.bind("<ButtonPress-1>", self.start_forward)  # Nhấn nút
        self.forward_button.bind("<ButtonRelease-1>", self.stop_movement)  # Nhả nút

        # Nút Lùi
        self.backward_button = tk.Button(self.root, text="Lùi")
        self.backward_button.pack(pady=10)
        self.backward_button.bind("<ButtonPress-1>", self.start_reverse)  # Nhấn nút
        self.backward_button.bind("<ButtonRelease-1>", self.stop_movement)  # Nhả nút

        # Kiểm tra hàng đợi định kỳ
        self.root.after(100, self.check_queue)

    def send_command(self, command):
        self.lidar_data.send_command(command)

    def start_forward(self, event):
        self.send_command("forward")  # Gửi lệnh forward một lần

    def start_reverse(self, event):
        self.send_command("reverse")  # Gửi lệnh reverse một lần

    def stop_movement(self, event):
        self.send_command("stop")  # Gửi lệnh stop khi nhả nút

    def check_queue(self):
        try:
            if not self.lidar_data.plot_queue.empty():
                self.lidar_data._plot_map()  # Gọi từ luồng chính
                self.lidar_data.plot_queue.get()  # Xóa tín hiệu khỏi hàng đợi
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        sensor = LidarData(host='192.168.1.226', port=80)
        gui = ControlGUI(sensor)
        data_thread = threading.Thread(target=sensor.update_data, daemon=True)
        data_thread.start()
        gui.run()
    except Exception as e:
        print(f"Program failed: {e}")