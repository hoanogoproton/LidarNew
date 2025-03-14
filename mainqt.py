import sys
import socket
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import pi
from PyQt5.QtWidgets import QApplication
import logging

from lidar_ui import LidarWindow  # Nhập giao diện từ file lidar_ui.py

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 50  # mm
    MAX_DATA_SIZE = 200  # Tích lũy nhiều điểm trước khi vẽ
    NEIGHBOR_RADIUS = 48
    MIN_NEIGHBORS = 4
    GRID_SIZE = 50  # mm

    def __init__(self, host='192.168.100.148', port=80, neighbor_radius=48, min_neighbors=4):
        self.host = host
        self.port = port
        self.sock = None
        self.data = {
            'angles': [],
            'distances': [],
            'speed': [],
            'x_coords': [],
            'y_coords': [],
        }
        self.grid = None
        self.robot_distance = 0.0  # Tổng quãng đường đã đi (mm)
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.plot_queue = queue.Queue()

        # Các tham số cấu hình
        self.NEIGHBOR_RADIUS = neighbor_radius
        self.MIN_NEIGHBORS = min_neighbors

        # Các đối tượng đồng bộ: Lock và Event
        self.data_lock = threading.Lock()
        self.data_event = threading.Event()
        self.running = True  # Cờ điều khiển vòng lặp của các thread

        # Thông số tính khoảng cách từ encoder (đơn vị cm)
        self.wheel_diameter = 60  # mm
        self.ppr = 200  # pulses per revolution
        self.pi = 3.1416
        self.wheel_circumference = self.wheel_diameter * self.pi  # mm

        # Các biến định vị (pose) của robot, lưu theo đơn vị mm
        self.pose_x = 0.0  # mm
        self.pose_y = 0.0  # mm
        self.pose_theta = 0.0  # radian

        # Các biến sai số cho bộ lọc (Kalman Filter placeholder)
        self.pose_x_error = 1.0
        self.pose_y_error = 1.0
        self.pose_theta_error = 2.5

        self.last_encoder_left = None
        self.last_encoder_right = None
        self.wheel_base = 50  # mm

        # Tạo figure cho matplotlib để hiển thị bản đồ
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("LiDAR Occupancy Grid Map with Robot Position")
        self.ax.grid(True)

        # Thử kết nối ban đầu
        if not self._connect_wifi():
            logging.error("Kết nối ban đầu không thành công. Vui lòng nhập IP ESP32 thủ công qua giao diện.")
        else:
            # Gửi lệnh reset encoders sau khi kết nối thành công
            self.send_command("RESET_ENCODERS")
            logging.info("Đã gửi lệnh reset encoders")

        # Khởi tạo các thread xử lý lệnh và dữ liệu
        self.command_thread = threading.Thread(target=self._handle_commands, daemon=True)
        self.command_thread.start()
        self.process_thread = threading.Thread(target=self.process_data, daemon=True)
        self.process_thread.start()

    def _connect_wifi(self):
        try:
            logging.info("Attempting to connect to %s:%s...", self.host, self.port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            logging.info("Connected to %s:%s", self.host, self.port)
            return True
        except (socket.error, socket.timeout) as e:
            logging.error("WiFi connection error: %s", e)
            return False

    def reconnect(self, new_host):
        self.host = new_host
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logging.error("Error closing socket: %s", e)
            self.sock = None
        if not self._connect_wifi():
            logging.error("Reconnection to %s failed.", new_host)
            return False
        else:
            logging.info("Reconnected successfully to %s.", new_host)
            return True

    # --------------------------
    # Các hàm xử lý dữ liệu cảm biến
    # --------------------------
    def _filter_data(self, angles, distances):
        # Sử dụng NumPy để lọc nhanh
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        mask = (distances_np >= self.MIN_DISTANCE) & (distances_np <= self.MAX_DISTANCE)
        return angles_np[mask].tolist(), distances_np[mask].tolist()

    def _to_cartesian(self, angles, distances):
        # Chuyển đổi vector hoá từ cực sang Cartesian
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        x_coords = distances_np * np.cos(angles_np)
        y_coords = distances_np * np.sin(angles_np)
        return x_coords.tolist(), y_coords.tolist()

    def _to_global_coordinates(self, angles, distances):
        # Chuyển vector hoá từ hệ sensor sang hệ toàn cục dựa trên pose hiện tại
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        global_x = self.pose_x + distances_np * np.cos(angles_np + self.pose_theta)
        global_y = self.pose_y + distances_np * np.sin(angles_np + self.pose_theta)
        return global_x.tolist(), global_y.tolist()

    def _remove_outliers(self, x_coords, y_coords):
        if len(x_coords) < self.MIN_NEIGHBORS:
            return [], []
        points = np.array(list(zip(x_coords, y_coords)))
        tree = cKDTree(points)

        # Tính khoảng cách đến MIN_NEIGHBORS hàng xóm gần nhất cho mỗi điểm
        distances, _ = tree.query(points, k=self.MIN_NEIGHBORS)
        # Lấy khoảng cách đến hàng xóm thứ MIN_NEIGHBORS
        distances_to_kth_neighbor = distances[:, self.MIN_NEIGHBORS - 1]
        # Tính trung bình và độ lệch chuẩn của khoảng cách
        mean_distance = np.mean(distances_to_kth_neighbor)
        std_distance = np.std(distances_to_kth_neighbor)
        # Đặt dynamic_radius dựa trên trung bình và độ lệch chuẩn
        dynamic_radius = mean_distance + 2 * std_distance

        # Đếm số hàng xóm với bán kính động
        neighbor_counts = tree.query_ball_point(points, r=dynamic_radius, return_length=True)
        mask = neighbor_counts >= self.MIN_NEIGHBORS

        # Lấy các điểm đã lọc sơ bộ
        filtered_points = points[mask]

        # Áp dụng lọc thống kê: loại bỏ các điểm có khoảng cách bất thường
        distances_from_origin = np.linalg.norm(filtered_points, axis=1)
        mean_dist = np.mean(distances_from_origin)
        std_dist = np.std(distances_from_origin)
        stat_mask = distances_from_origin <= (mean_dist + 2 * std_dist)
        final_points = filtered_points[stat_mask]

        return final_points[:, 0].tolist(), final_points[:, 1].tolist()

    def create_occupancy_grid(self):
        grid_size = int(2 * self.MAX_DISTANCE / self.GRID_SIZE)
        if self.grid is None:
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        with self.data_lock:
            for x, y in zip(self.data['x_coords'], self.data['y_coords']):
                grid_x = int((x + self.MAX_DISTANCE) / self.GRID_SIZE)
                grid_y = int((y + self.MAX_DISTANCE) / self.GRID_SIZE)
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    self.grid[grid_x, grid_y] = 255
        return self.grid

    def _plot_map(self):
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return

        global_x_coords, global_y_coords = self._to_global_coordinates(angles, distances)
        filtered_x, filtered_y = self._remove_outliers(global_x_coords, global_y_coords)

        with self.data_lock:
            self.data['x_coords'].extend(filtered_x)
            self.data['y_coords'].extend(filtered_y)
            self.data['angles'].clear()
            self.data['distances'].clear()
        grid = self.create_occupancy_grid()
        self.ax.clear()
        self.ax.imshow(grid, cmap='gray', origin='lower',
                       extent=(-self.MAX_DISTANCE, self.MAX_DISTANCE, -self.MAX_DISTANCE, self.MAX_DISTANCE))

        # Vẽ vị trí của robot
        self.ax.plot(self.pose_x, self.pose_y, 'ro', markersize=10)

        # Tính toán điểm kết thúc của đường thẳng biểu diễn hướng
        arrow_length = 500  # Độ dài đường thẳng (mm), bạn có thể điều chỉnh
        end_x = self.pose_x + arrow_length * np.cos(self.pose_theta)
        end_y = self.pose_y + arrow_length * np.sin(self.pose_theta)

        # Vẽ đường thẳng biểu diễn hướng
        self.ax.plot([self.pose_x, end_x], [self.pose_y, end_y], 'r-', linewidth=2)

        self.ax.set_title(f"LiDAR Occupancy Grid Map - Distance: {self.robot_distance:.2f} mm")
        self.ax.grid(True)
        self.fig.canvas.draw()

    def process_data(self):
        while self.running:
            if self.data_event.wait(timeout=0.1):
                # Khi có dữ liệu mới, chuyển tiếp đến plot_queue
                while not self.data_queue.empty():
                    self.plot_queue.put(True)
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
                self.data_event.clear()

    def update_data(self):
        buffer = ""
        while self.running:
            if self.sock is None:
                time.sleep(0.1)
                continue
            try:
                data = self.sock.recv(4096)
                if not data:
                    logging.warning("Connection closed by server.")
                    self.sock = None
                    continue
                data = data.decode('utf-8', errors='ignore')
                buffer += data
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except Exception as e:
                logging.error("Exception in update_data: %s", e)
                self.sock = None
                continue

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                logging.debug("Raw data received: %s", line)
                sensor_data = line.strip().split('\t')
                if len(sensor_data) != 9:
                    logging.warning("Sensor data length mismatch: %s", sensor_data)
                    continue

                try:
                    base_angle = int(sensor_data[0])
                    speed = int(sensor_data[1])
                    distances = np.array(sensor_data[2:6], dtype=float)
                    encoder_count = int(sensor_data[7].strip())
                    encoder_count2 = int(sensor_data[8].strip())

                    # Log giá trị encoder thô ngay khi nhận được
                    logging.info("Raw encoder values: encoder_count=%d, encoder_count2=%d",
                                 encoder_count, encoder_count2)
                except ValueError as e:
                    logging.error("Error parsing sensor data %s: %s", sensor_data, e)
                    continue

                # --- Xử lý odometry ---
                if self.last_encoder_left is None:
                    # Lần đọc đầu tiên: gán giá trị encoder ban đầu và reset robot_distance về 0
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2
                    self.initial_encoder_left = encoder_count
                    self.initial_encoder_right = encoder_count2
                    self.robot_distance = 0.0
                    logging.info("Initialized encoders: left=%d, right=%d",
                                 encoder_count, encoder_count2)
                else:
                    delta_left = (encoder_count - self.last_encoder_left) * self.wheel_circumference / self.ppr  # mm
                    delta_right = (encoder_count2 - self.last_encoder_right) * self.wheel_circumference / self.ppr  # mm
                    delta_s = (delta_left + delta_right) / 2.0  # mm
                    delta_theta = (delta_right - delta_left) / self.wheel_base  # radian
                    delta_s_mm = delta_s  # Đã tính bằng mm

                    # Log giá trị trung gian
                    logging.debug("Encoder: left=%d, right=%d, delta_left=%.2f mm, delta_right=%.2f mm",
                                  encoder_count, encoder_count2, delta_left, delta_right)
                    logging.debug("Odometry: delta_s=%.2f mm, delta_theta=%.4f rad", delta_s, delta_theta)

                    new_pose_x = self.pose_x + delta_s_mm * np.cos(self.pose_theta + delta_theta / 2)
                    new_pose_y = self.pose_y + delta_s_mm * np.sin(self.pose_theta + delta_theta / 2)
                    new_pose_theta = self.pose_theta + delta_theta

                    self.pose_x, self.pose_x_error = self.kalman_filter(
                        new_pose_x, self.pose_x, self.pose_x_error, process_variance=1e-3, measurement_variance=1e-2)
                    self.pose_y, self.pose_y_error = self.kalman_filter(
                        new_pose_y, self.pose_y, self.pose_y_error, process_variance=1e-3, measurement_variance=1e-2)
                    self.pose_theta, self.pose_theta_error = self.kalman_filter(
                        new_pose_theta, self.pose_theta, self.pose_theta_error, process_variance=1e-3,
                        measurement_variance=1e-2)

                    # Log pose sau khi cập nhật
                    logging.info("Pose updated: x=%.2f mm, y=%.2f mm, theta=%.4f rad",
                                 self.pose_x, self.pose_y, self.pose_theta)

                    # Log giá trị encoder trước và sau khi cập nhật
                    logging.debug("Encoder update: prev_left=%d, prev_right=%d, new_left=%d, new_right=%d",
                                  self.last_encoder_left, self.last_encoder_right, encoder_count, encoder_count2)

                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2

                # Cập nhật tổng quãng đường di chuyển
                self.robot_distance = (((encoder_count - self.initial_encoder_left) + (
                        encoder_count2 - self.initial_encoder_right)) / 2
                                       * self.wheel_circumference / self.ppr)
                logging.info("Total distance traveled: %.2f mm", self.robot_distance)

                # --- Xử lý dữ liệu LiDAR ---
                angles = (np.arange(4) + base_angle) * (pi / 180)
                valid_mask = (distances >= self.MIN_DISTANCE) & (distances <= self.MAX_DISTANCE)
                valid_angles = angles[valid_mask]
                valid_distances = distances[valid_mask]

                with self.data_lock:
                    self.data['angles'].extend(valid_angles.tolist())
                    self.data['distances'].extend(valid_distances.tolist())
                    self.data['speed'].extend([speed] * int(np.sum(valid_mask)))
                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self.data_queue.put(True)
                        self.data_event.set()
                    if len(self.data['angles']) > 500:
                        indices = np.random.choice(len(self.data['angles']), 200, replace=False)
                        self.data['angles'] = [self.data['angles'][i] for i in indices]
                        self.data['distances'] = [self.data['distances'][i] for i in indices]
                        self.data['speed'] = [self.data['speed'][i] for i in indices]

    # --------------------------
    # Placeholder cho bộ lọc nhiễu: Kalman Filter đơn giản
    # --------------------------
    def kalman_filter(self, measurement, prev_estimate, prev_error, process_variance, measurement_variance):
        """
        Một phiên bản rất đơn giản của Kalman Filter.
        measurement: giá trị đo được hiện tại.
        prev_estimate: ước tính từ bước trước.
        prev_error: sai số của ước tính trước.
        process_variance: phương sai của hệ thống.
        measurement_variance: phương sai của đo đạc.
        """
        # Tính toán hệ số Kalman
        K = prev_error / (prev_error + measurement_variance)
        # Cập nhật ước tính
        estimate = prev_estimate + K * (measurement - prev_estimate)
        # Cập nhật sai số ước tính
        error = (1 - K) * prev_error + abs(prev_estimate - measurement) * process_variance
        return estimate, error

    def send_command(self, command):
        self.command_queue.put(command)

    def _handle_commands(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if self.sock:
                    self.sock.sendall((command + '\n').encode('utf-8'))
                    logging.info("Sent command: %s", command)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error("Failed to send command: %s", e)
                time.sleep(0.01)

    def get_coordinates(self):
        with self.data_lock:
            return self.data['x_coords'][:], self.data['y_coords'][:]

    def cleanup(self):
        self.running = False
        self.data_event.set()
        if self.sock:
            self.sock.close()
            logging.info("WiFi connection closed.")
        plt.close(self.fig)
        logging.info("Plot closed.")


if __name__ == "__main__":
    lidar = LidarData(host='192.168.100.148', port=80, neighbor_radius=50, min_neighbors=5)
    data_thread = threading.Thread(target=lidar.update_data, daemon=True)
    data_thread.start()

    app = QApplication(sys.argv)
    window = LidarWindow(lidar)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        lidar.cleanup()
