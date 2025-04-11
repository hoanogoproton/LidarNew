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
import heapq

from lidar_ui import LidarWindow  # Nhập giao diện từ file lidar_ui.py

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def heuristic(a, b):
    # Sử dụng khoảng cách Euclid
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_priority, current = heapq.heappop(open_set)
        if current == goal:
            break
        x, y = current
        # 8 hướng di chuyển
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                     (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
        for next in neighbors:
            nx, ny = next
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] != 255:
                # Sử dụng √2 cho bước chéo, 1 cho bước ngang/dọc
                step_cost = sqrt(2) if (nx - x != 0 and ny - y != 0) else 1
                new_cost = cost_so_far[current] + step_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    heapq.heappush(open_set, (priority, next))
                    came_from[next] = current

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []  # Không tìm được đường đi
    path.append(start)
    path.reverse()
    return path

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

def is_collision_free(p1, p2, grid):
    """
    Kiểm tra đường thẳng từ p1 đến p2 trong occupancy grid có va chạm với chướng ngại vật hay không.
    p1, p2: tuple (x, y) của chỉ số ô (grid coordinates)
    grid: occupancy grid với 0 là free, 255 là obstacle.
    Giả sử các ô có giá trị 127 (unknown) cũng được coi là free.
    """
    cells = bresenham_line(p1[0], p1[1], p2[0], p2[1])
    for cell in cells:
        x, y = cell
        # Nếu ô có giá trị 255 (occupied) thì coi là va chạm
        if grid[y, x] == 255:
            return False
    return True

def smooth_path(path, grid):
    if not path or len(path) < 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if is_collision_free(smoothed[-1], path[j], grid):
                smoothed.append(path[j])
                i = j
                break
        else:
            # Nếu không tìm được điểm nào xa hơn, thêm điểm tiếp theo
            smoothed.append(path[i + 1])
            i += 1
    return smoothed
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
        theta = self.state[2]
        dx = delta_s * cos(theta + delta_theta / 2)
        dy = delta_s * sin(theta + delta_theta / 2)
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += delta_theta
        self.state[2] = atan2(sin(self.state[2]), cos(self.state[2]))
        F = np.array([
            [1, 0, -delta_s * sin(theta + delta_theta / 2)],
            [0, 1,  delta_s * cos(theta + delta_theta / 2)],
            [0, 0, 1]
        ])
        self.cov = F @ self.cov @ F.T + motion_cov

    def update(self, z, landmark_pos, measurement_cov):
        dx = landmark_pos[0] - self.state[0]
        dy = landmark_pos[1] - self.state[1]
        q = dx**2 + dy**2
        expected_range = sqrt(q)
        expected_bearing = atan2(dy, dx) - self.state[2]
        z_hat = np.array([expected_range, expected_bearing])
        y = z - z_hat
        y[1] = atan2(sin(y[1]), cos(y[1]))
        H = np.array([
            [-dx / expected_range, -dy / expected_range, 0],
            [dy / q,              -dx / q,             -1]
        ])
        S = H @ self.cov @ H.T + measurement_cov
        K = self.cov @ H.T @ np.linalg.inv(S)
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
    GRID_SIZE = 30      # mm

    def __init__(self, host='192.168.100.148', port=80, neighbor_radius=48, min_neighbors=4):
        self.host = host
        self.port = port
        self.sock = None
        self.data = {
            'angles': [], 'distances': [], 'speed': [],
            'x_coords': [], 'y_coords': []
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

        # Các thuộc tính khác
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        # Khởi tạo ekf_slam ngay từ đầu
        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])

        # Thông số encoder
        self.wheel_diameter = 70  # mm
        self.ppr = 500            # pulses per revolution
        self.pi = 3.1416
        self.wheel_circumference = self.wheel_diameter * self.pi  # mm

        # Biến định vị (pose) của robot: [x, y, theta]
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.heading_offset = 0.0  # Offset góc ban đầu là 0 (radian)

        self.last_encoder_left = None
        self.last_encoder_right = None
        self.wheel_base = 138  # mm

        # Khởi tạo figure cho matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Global Map with Robot Position")
        self.ax.grid(True)

        # Khởi tạo global map (occupancy grid)
        self.map_size_mm = 10000  # mm, tức là 10m x 10m
        self.global_map_dim = int(self.map_size_mm / self.GRID_SIZE)
        self.global_map = np.full((self.global_map_dim, self.global_map_dim), 127, dtype=np.uint8)

        self.pose_lock = threading.Lock()
        if not self._connect_wifi():
            logging.error("Kết nối ban đầu không thành công. Vui lòng nhập IP ESP32 thủ công qua giao diện.")
        else:
            self.send_command("RESET_ENCODERS")
            logging.info("Đã gửi lệnh reset encoders")

        self.command_thread = threading.Thread(target=self._handle_commands, daemon=True)
        self.command_thread.start()
        self.process_thread = threading.Thread(target=self.process_data, daemon=True)
        self.process_thread.start()

    def set_heading(self, theta):
        with self.pose_lock:
            self.pose_theta = theta
            if hasattr(self, 'ekf_slam'):
                self.ekf_slam.state[2] = theta
        logging.info(f"Heading adjusted to θ={theta:.3f} rad")

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
        angles_np = np.array(angles) + self.heading_offset
        distances_np = np.array(distances)
        global_x = self.pose_x + distances_np * np.cos(self.pose_theta + angles_np)
        global_y = self.pose_y + distances_np * np.sin(self.pose_theta + angles_np)
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
        robot_grid_x = int((self.pose_x + self.map_size_mm/2) / self.GRID_SIZE)
        robot_grid_y = int((self.pose_y + self.map_size_mm/2) / self.GRID_SIZE)
        for x, y in zip(scan_x, scan_y):
            meas_grid_x = int((x + self.map_size_mm/2) / self.GRID_SIZE)
            meas_grid_y = int((y + self.map_size_mm/2) / self.GRID_SIZE)
            line_cells = bresenham_line(robot_grid_x, robot_grid_y, meas_grid_x, meas_grid_y)
            for cell in line_cells[:-1]:
                cx, cy = cell
                if 0 <= cx < self.global_map_dim and 0 <= cy < self.global_map_dim:
                    self.global_map[cy, cx] = 0
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
                if len(sensor_data) != 10:
                    logging.warning("Sensor data length mismatch: %s", sensor_data)
                    continue
                try:
                    base_angle = int(sensor_data[0])
                    speed = int(sensor_data[1])
                    distances = np.array(sensor_data[2:6], dtype=float)
                    encoder_count = int(sensor_data[7].strip())
                    encoder_count2 = int(sensor_data[8].strip())
                    gyro_z = float(sensor_data[9].strip())
                    logging.info("Raw data: encoder_count=%d, encoder_count2=%d, gyroZ=%.2f rad/s",
                                 encoder_count, encoder_count2, gyro_z)
                except ValueError as e:
                    logging.error("Error parsing sensor data %s: %s", sensor_data, e)
                    continue
                if self.last_encoder_left is None:
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2
                    self.initial_encoder_left = encoder_count
                    self.initial_encoder_right = encoder_count2
                    self.robot_distance = 0.0
                    self.last_time = time.time()
                    logging.info("Initialized encoders: left=%d, right=%d", encoder_count, encoder_count2)
                else:
                    current_time = time.time()
                    delta_t = current_time - self.last_time
                    self.last_time = current_time
                    delta_left = (encoder_count - self.last_encoder_left) * self.wheel_circumference / self.ppr
                    delta_right = (encoder_count2 - self.last_encoder_right) * self.wheel_circumference / self.ppr
                    delta_s = (delta_left + delta_right) / 2.0
                    delta_theta_enc = (delta_right - delta_left) / self.wheel_base
                    delta_theta_gyro = gyro_z * delta_t
                    delta_theta = 1 * delta_theta_gyro + 0 * delta_theta_enc
                    logging.debug("Encoder: left=%d, right=%d, delta_left=%.2f mm, delta_right=%.2f mm",
                                  encoder_count, encoder_count2, delta_left, delta_right)
                    logging.debug("Odometry: delta_s=%.2f mm, delta_theta_enc=%.4f rad, delta_theta_gyro=%.4f rad",
                                  delta_s, delta_theta_enc, delta_theta_gyro)
                    if not hasattr(self, 'ekf_slam'):
                        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])
                    motion_cov = np.diag([1e-1, 1e-1, 1e-2])
                    self.ekf_slam.predict(delta_s, delta_theta, motion_cov)
                    self.pose_x, self.pose_y, self.pose_theta = self.ekf_slam.state
                    logging.info("Pose updated (EKF): x=%.2f mm, y=%.2f mm, theta=%.4f rad",
                                 self.pose_x, self.pose_y, self.pose_theta)
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2
                self.robot_distance = (((encoder_count - self.initial_encoder_left) +
                                        (encoder_count2 - self.initial_encoder_right)) / 2
                                       * self.wheel_circumference / self.ppr)
                logging.info("Total distance traveled: %.2f mm", self.robot_distance)
                angles = (np.arange(4) + base_angle) * (pi / 180)
                valid_mask = (distances >= self.MIN_DISTANCE) & (distances <= self.MAX_DISTANCE)
                valid_angles = angles[valid_mask]
                valid_distances = distances[valid_mask]
                if valid_angles.size > 0:
                    for i in range(len(valid_angles)):
                        meas_range = valid_distances[i]
                        meas_bearing = valid_angles[i] - self.pose_theta
                        z = np.array([meas_range, meas_bearing])
                        lx = self.ekf_slam.state[0] + meas_range * cos(self.ekf_slam.state[2] + meas_bearing)
                        ly = self.ekf_slam.state[1] + meas_range * sin(self.ekf_slam.state[2] + meas_bearing)
                        landmark_pos = np.array([lx, ly])
                        measurement_cov = np.diag([1e-3, 1e-4])
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

    def move(self, distance_m):
        # Lưu lại vị trí ban đầu (đơn vị mm)
        initial_x, initial_y = self.pose_x, self.pose_y
        # Gửi lệnh di chuyển với đơn vị là mét
        cmd = f"MOVE {distance_m}"
        self.send_command(cmd)

        tolerance = 0.02  # cho phép sai số 2cm
        start_time = time.time()
        timeout = 10  # thời gian tối đa chờ là 10 giây

        while True:
            # Chuyển đổi khoảng cách đã đi từ mm sang m để so sánh
            current_distance_m = sqrt((self.pose_x - initial_x) ** 2 + (self.pose_y - initial_y) ** 2) / 1000.0
            if current_distance_m >= distance_m - tolerance:
                break
            if time.time() - start_time > timeout:
                print("Timeout khi di chuyển!")
                break
            time.sleep(0.1)

    def rotate(self, angle_rad):
        # Tính góc mục tiêu theo radian dựa trên odometry hiện tại
        target_theta = self.pose_theta + angle_rad
        target_theta = atan2(sin(target_theta), cos(target_theta))
        tolerance = 0.1  # cho phép sai số 0.1 rad
        # Gửi lệnh quay với đơn vị radian
        cmd = f"ROTATE {angle_rad}"
        self.send_command(cmd)

        start_time = time.time()
        timeout = 10  # timeout 10 giây

        while True:
            # Tính sai số góc hiện tại (radian)
            error = abs(atan2(sin(self.pose_theta - target_theta), cos(self.pose_theta - target_theta)))
            if error < tolerance:
                break
            if time.time() - start_time > timeout:
                print("Timeout khi quay!")
                break
            time.sleep(0.1)
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

    def reset_map(self):
        self.global_map = np.full((self.global_map_dim, self.global_map_dim), 127, dtype=np.uint8)
        with self.data_lock:
            self.data['x_coords'].clear()
            self.data['y_coords'].clear()
        print("Đã xóa bản đồ.")

    # --- HÀM NAVIGATE_TO_TARGET ĐƯỢC THÊM VÀO LỚP ---
    def navigate_to_target(self, target_x, target_y):
        # Tính tọa độ đích trên lưới (grid)
        grid_target = (int((target_x + self.map_size_mm / 2) / self.GRID_SIZE),
                       int((target_y + self.map_size_mm / 2) / self.GRID_SIZE))
        grid_start = (int((self.pose_x + self.map_size_mm / 2) / self.GRID_SIZE),
                      int((self.pose_y + self.map_size_mm / 2) / self.GRID_SIZE))
        path = a_star_search(self.global_map, grid_start, grid_target)
        if not path:
            print("Không tìm được đường đi đến đích!")
            return

        smooth_waypoints = smooth_path(path, self.global_map)
        # Chuyển đổi tọa độ grid về tọa độ thực (mm)
        waypoints_x = [cell[0] * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2 for cell in
                       smooth_waypoints]
        waypoints_y = [cell[1] * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2 for cell in
                       smooth_waypoints]

        # Vẽ các waypoint lên bản đồ để theo dõi
        self.ax.plot(waypoints_x, waypoints_y, 'bo-', markersize=5, label='Smoothed Waypoints')
        self.ax.legend()
        self.fig.canvas.draw()

        # Di chuyển qua từng waypoint
        for wx, wy in zip(waypoints_x, waypoints_y):
            desired_angle = atan2(wy - self.pose_y, wx - self.pose_x)
            angle_diff = desired_angle - self.pose_theta
            angle_diff = atan2(sin(angle_diff), cos(angle_diff))
            print(f"Di chuyển đến waypoint tại ({wx:.1f}, {wy:.1f}); cần xoay {angle_diff:.3f} rad")
            self.rotate(angle_diff)
            # Tính khoảng cách cần di chuyển (đã ở mm)
            distance = sqrt((wx - self.pose_x) ** 2 + (wy - self.pose_y) ** 2)
            # Chuyển khoảng cách từ mm sang mét trước khi gọi move
            self.move(distance / 1000)
        print("Đã hoàn thành di chuyển đến vị trí đích")
# ------------------------------
# Chương trình chính
# ------------------------------
if __name__ == "__main__":
    lidar = LidarData(host='192.168.0.112', port=80, neighbor_radius=30, min_neighbors=7)
    data_thread = threading.Thread(target=lidar.update_data, daemon=True)
    data_thread.start()
    app = QApplication(sys.argv)
    window = LidarWindow(lidar)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        lidar.cleanup()