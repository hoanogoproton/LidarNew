import sys
import socket
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import pi, cos, sin, atan2, sqrt
from PyQt5.QtWidgets import QApplication
import logging

from lidar_ui import LidarWindow  # Nhập giao diện từ file lidar_ui.py

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ------------------------------
# Hàm Bresenham cho việc tìm các ô trong grid theo đường ray
# ------------------------------
def bresenham_line(x0, y0, x1, y1):
    """
    Trả về danh sách các ô (x, y) theo thuật toán Bresenham từ (x0, y0) đến (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


# ------------------------------
# Lớp EKF SLAM
# ------------------------------
class EKFSLAM:
    def __init__(self, initial_pose):
        # State vector gồm: [x, y, theta]
        self.state = np.array(initial_pose, dtype=float)
        # Ma trận hiệp phương sai khởi tạo
        self.cov = np.eye(3) * 0.1

    def predict(self, delta_s, delta_theta, motion_cov):
        """
        Dự đoán pose dựa trên mô hình chuyển động.
        delta_s: khoảng cách di chuyển (mm)
        delta_theta: thay đổi góc (radian)
        motion_cov: ma trận nhiễu của mô hình chuyển động
        """
        theta = self.state[2]
        # Cập nhật pose theo mô hình chuyển động
        dx = delta_s * cos(theta + delta_theta / 2)
        dy = delta_s * sin(theta + delta_theta / 2)
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += delta_theta
        # Chuẩn hóa góc về [-pi, pi]
        self.state[2] = atan2(sin(self.state[2]), cos(self.state[2]))

        # Tính jacobian của mô hình chuyển động theo state
        F = np.array([
            [1, 0, -delta_s * sin(theta + delta_theta / 2)],
            [0, 1,  delta_s * cos(theta + delta_theta / 2)],
            [0, 0, 1]
        ])
        # Cập nhật covariance
        self.cov = F @ self.cov @ F.T + motion_cov

    def update(self, z, landmark_pos, measurement_cov):
        """
        Cập nhật pose dựa trên đo đạc landmark.
        z: đo đạc [range, bearing] (theo hệ tọa độ của robot)
        landmark_pos: vị trí landmark trong hệ tọa độ toàn cục (giả sử đã biết)
        measurement_cov: ma trận nhiễu của đo đạc
        """
        dx = landmark_pos[0] - self.state[0]
        dy = landmark_pos[1] - self.state[1]
        q = dx**2 + dy**2
        expected_range = sqrt(q)
        expected_bearing = atan2(dy, dx) - self.state[2]
        z_hat = np.array([expected_range, expected_bearing])

        # Tính sai lệch đo đạc (innovation)
        y = z - z_hat
        y[1] = atan2(sin(y[1]), cos(y[1]))  # chuẩn hóa góc

        # Tính jacobian của phép đo theo state
        H = np.array([
            [-dx / expected_range, -dy / expected_range, 0],
            [dy / q,              -dx / q,             -1]
        ])
        S = H @ self.cov @ H.T + measurement_cov
        K = self.cov @ H.T @ np.linalg.inv(S)

        # Cập nhật state và covariance
        self.state = self.state + K @ y
        self.cov = (np.eye(3) - K @ H) @ self.cov


# ------------------------------
# Lớp xử lý dữ liệu LiDAR, odometry và xây dựng bản đồ toàn cục
# ------------------------------
class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 50    # mm
    MAX_DATA_SIZE = 180  # Tích lũy nhiều điểm trước khi vẽ
    NEIGHBOR_RADIUS = 48
    MIN_NEIGHBORS = 4
    GRID_SIZE = 20      # mm

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

        self.NEIGHBOR_RADIUS = neighbor_radius
        self.MIN_NEIGHBORS = min_neighbors

        self.data_lock = threading.Lock()
        self.data_event = threading.Event()
        self.running = True

        # Thông số encoder
        self.wheel_diameter = 70  # mm
        self.ppr = 695            # pulses per revolution
        self.pi = 3.1416
        self.wheel_circumference = self.wheel_diameter * self.pi  # mm

        # Biến định vị (pose) của robot: [x, y, theta]
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0

        self.last_encoder_left = None
        self.last_encoder_right = None
        self.wheel_base = 138  # mm

        # Khởi tạo figure cho matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Global Map with Robot Position")
        self.ax.grid(True)

        # Khởi tạo global map (occupancy grid)
        self.map_size_mm = 10000  # Kích thước map toàn cục (10m x 10m)
        self.global_map_dim = int(self.map_size_mm / self.GRID_SIZE)
        # Khởi tạo với giá trị 127 (unknown); giá trị 0: free, 255: occupied
        self.global_map = np.full((self.global_map_dim, self.global_map_dim), 127, dtype=np.uint8)

        # Kết nối ban đầu
        if not self._connect_wifi():
            logging.error("Kết nối ban đầu không thành công. Vui lòng nhập IP ESP32 thủ công qua giao diện.")
        else:
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
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        mask = (distances_np >= self.MIN_DISTANCE) & (distances_np <= self.MAX_DISTANCE)
        return angles_np[mask].tolist(), distances_np[mask].tolist()

    def _to_cartesian(self, angles, distances):
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        x_coords = distances_np * np.cos(angles_np)
        y_coords = distances_np * np.sin(angles_np)
        return x_coords.tolist(), y_coords.tolist()

    def _to_global_coordinates(self, angles, distances):
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
        distances, _ = tree.query(points, k=self.MIN_NEIGHBORS)
        distances_to_kth_neighbor = distances[:, self.MIN_NEIGHBORS - 1]
        mean_distance = np.mean(distances_to_kth_neighbor)
        std_distance = np.std(distances_to_kth_neighbor)
        dynamic_radius = mean_distance + 2 * std_distance
        neighbor_counts = tree.query_ball_point(points, r=dynamic_radius, return_length=True)
        mask = neighbor_counts >= self.MIN_NEIGHBORS
        filtered_points = points[mask]
        distances_from_origin = np.linalg.norm(filtered_points, axis=1)
        mean_dist = np.mean(distances_from_origin)
        std_dist = np.std(distances_from_origin)
        stat_mask = distances_from_origin <= (mean_dist + 2 * std_dist)
        final_points = filtered_points[stat_mask]
        return final_points[:, 0].tolist(), final_points[:, 1].tolist()

    def voxel_downsample(self, x_coords, y_coords, voxel_size=20):
        voxel_dict = {}
        for x, y in zip(x_coords, y_coords):
            voxel_x = int(x // voxel_size)
            voxel_y = int(y // voxel_size)
            key = (voxel_x, voxel_y)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append((x, y))
        downsampled_x = []
        downsampled_y = []
        for points in voxel_dict.values():
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            downsampled_x.append(avg_x)
            downsampled_y.append(avg_y)
        return downsampled_x, downsampled_y

    # --------------------------
    # Cập nhật global map bằng thuật toán map merging
    # --------------------------
    def update_global_map(self, scan_x, scan_y):
        """
        Cập nhật global_map từ các điểm quét (scan_x, scan_y) với tọa độ toàn cục (mm).
        Sử dụng thuật toán Bresenham để xác định các ô “free” theo đường ray từ vị trí robot đến điểm đo.
        """
        # Chuyển vị trí robot (mm) sang chỉ số lưới global map
        robot_grid_x = int((self.pose_x + self.map_size_mm/2) / self.GRID_SIZE)
        robot_grid_y = int((self.pose_y + self.map_size_mm/2) / self.GRID_SIZE)
        for x, y in zip(scan_x, scan_y):
            meas_grid_x = int((x + self.map_size_mm/2) / self.GRID_SIZE)
            meas_grid_y = int((y + self.map_size_mm/2) / self.GRID_SIZE)
            line_cells = bresenham_line(robot_grid_x, robot_grid_y, meas_grid_x, meas_grid_y)
            # Cập nhật các ô dọc theo đường ray là free (trừ ô cuối cùng)
            for cell in line_cells[:-1]:
                cx, cy = cell
                if 0 <= cx < self.global_map_dim and 0 <= cy < self.global_map_dim:
                    self.global_map[cy, cx] = 0
            # Ô cuối cùng đánh dấu là occupied
            if 0 <= meas_grid_x < self.global_map_dim and 0 <= meas_grid_y < self.global_map_dim:
                self.global_map[meas_grid_y, meas_grid_x] = 255

    def _plot_map(self):
        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if not angles:
            return

        global_x_coords, global_y_coords = self._to_global_coordinates(angles, distances)
        filtered_x, filtered_y = self._remove_outliers(global_x_coords, global_y_coords)
        downsampled_x, downsampled_y = self.voxel_downsample(filtered_x, filtered_y, voxel_size=self.GRID_SIZE)

        with self.data_lock:
            self.data['x_coords'].extend(downsampled_x)
            self.data['y_coords'].extend(downsampled_y)
            self.data['angles'].clear()
            self.data['distances'].clear()

        # Cập nhật global map từ các điểm quét mới
        self.update_global_map(downsampled_x, downsampled_y)

        self.ax.clear()
        extent = (-self.map_size_mm/2, self.map_size_mm/2, -self.map_size_mm/2, self.map_size_mm/2)
        self.ax.imshow(self.global_map, cmap='gray', origin='lower', extent=extent)
        self.ax.plot(self.pose_x, self.pose_y, 'ro', markersize=10)
        arrow_length = 500  # mm
        end_x = self.pose_x + arrow_length * cos(self.pose_theta)
        end_y = self.pose_y + arrow_length * sin(self.pose_theta)
        self.ax.plot([self.pose_x, end_x], [self.pose_y, end_y], 'r-', linewidth=2)
        self.ax.set_title(f"Global Map - Total Distance: {self.robot_distance:.2f} mm")
        self.ax.grid(True)
        self.fig.canvas.draw()

    def process_data(self):
        while self.running:
            if self.data_event.wait(timeout=0.1):
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
                    logging.info("Raw encoder values: encoder_count=%d, encoder_count2=%d",
                                 encoder_count, encoder_count2)
                except ValueError as e:
                    logging.error("Error parsing sensor data %s: %s", sensor_data, e)
                    continue

                # --- Xử lý odometry ---
                if self.last_encoder_left is None:
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2
                    self.initial_encoder_left = encoder_count
                    self.initial_encoder_right = encoder_count2
                    self.robot_distance = 0.0
                    logging.info("Initialized encoders: left=%d, right=%d", encoder_count, encoder_count2)
                else:
                    delta_left = (encoder_count - self.last_encoder_left) * self.wheel_circumference / self.ppr  # mm
                    delta_right = (encoder_count2 - self.last_encoder_right) * self.wheel_circumference / self.ppr  # mm
                    delta_s = (delta_left + delta_right) / 2.0  # mm
                    delta_theta = (delta_right - delta_left) / self.wheel_base  # radian

                    logging.debug("Encoder: left=%d, right=%d, delta_left=%.2f mm, delta_right=%.2f mm",
                                  encoder_count, encoder_count2, delta_left, delta_right)
                    logging.debug("Odometry: delta_s=%.2f mm, delta_theta=%.4f rad", delta_s, delta_theta)

                    # --- EKF SLAM Prediction ---
                    if not hasattr(self, 'ekf_slam'):
                        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])
                    motion_cov = np.diag([1e-3, 1e-3, 1e-4])
                    self.ekf_slam.predict(delta_s, delta_theta, motion_cov)

                    # Cập nhật pose từ EKF
                    self.pose_x, self.pose_y, self.pose_theta = self.ekf_slam.state
                    logging.info("Pose updated (EKF): x=%.2f mm, y=%.2f mm, theta=%.4f rad",
                                 self.pose_x, self.pose_y, self.pose_theta)

                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2

                self.robot_distance = (((encoder_count - self.initial_encoder_left) +
                                        (encoder_count2 - self.initial_encoder_right)) / 2
                                       * self.wheel_circumference / self.ppr)
                logging.info("Total distance traveled: %.2f mm", self.robot_distance)

                # --- Xử lý dữ liệu LiDAR ---
                angles = (np.arange(4) + base_angle) * (pi / 180)
                valid_mask = (distances >= self.MIN_DISTANCE) & (distances <= self.MAX_DISTANCE)
                valid_angles = angles[valid_mask]
                valid_distances = distances[valid_mask]

                # Ví dụ: sử dụng đo đạc đầu tiên làm landmark update
                if valid_angles.size > 0:
                    if not hasattr(self, 'ekf_slam'):
                        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])
                    meas_range = valid_distances[0]
                    meas_bearing = valid_angles[0] - self.pose_theta  # Góc đo tương đối
                    z = np.array([meas_range, meas_bearing])
                    lx = self.ekf_slam.state[0] + meas_range * cos(self.ekf_slam.state[2] + meas_bearing)
                    ly = self.ekf_slam.state[1] + meas_range * sin(self.ekf_slam.state[2] + meas_bearing)
                    landmark_pos = np.array([lx, ly])
                    measurement_cov = np.diag([1e-2, 1e-3])
                    self.ekf_slam.update(z, landmark_pos, measurement_cov)

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


# ------------------------------
# Chương trình chính
# ------------------------------
if __name__ == "__main__":
    lidar = LidarData(host='192.168.0.113', port=80, neighbor_radius=50, min_neighbors=5)
    data_thread = threading.Thread(target=lidar.update_data, daemon=True)
    data_thread.start()

    app = QApplication(sys.argv)
    window = LidarWindow(lidar)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        lidar.cleanup()
