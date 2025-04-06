from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSlider
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from math import pi
from PyQt5.QtWidgets import QMessageBox


class LidarWindow(QMainWindow):
    def __init__(self, lidar):
        super().__init__()
        self.lidar = lidar
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Robot Control with Lidar Mapping")
        self.setGeometry(100, 100, 800, 600)
        # Đảm bảo cửa sổ nhận được sự kiện bàn phím
        self.setFocusPolicy(Qt.StrongFocus)

        # Tạo widget trung tâm và layout chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout(central_widget)

        # Tích hợp Matplotlib canvas từ figure của LidarData
        self.canvas = FigureCanvas(self.lidar.fig)
        vbox.addWidget(self.canvas)

        # Layout cho cài đặt kết nối: nhập địa chỉ IP ESP32 và nút kết nối
        conn_layout = QHBoxLayout()
        self.ip_label = QLabel("ESP32 IP:")
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Nhập địa chỉ IP")
        self.connect_btn = QPushButton("Kết nối")
        self.connect_btn.clicked.connect(self.reconnect_device)
        conn_layout.addWidget(self.ip_label)
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.connect_btn)
        vbox.addLayout(conn_layout)

        # ── Layout cho chỉnh hướng ban đầu ──
        heading_layout = QHBoxLayout()
        self.heading_input = QLineEdit()
        self.heading_input.setPlaceholderText("Θ (deg)")
        self.set_heading_btn = QPushButton("Set Heading")
        self.set_heading_btn.clicked.connect(self.on_set_heading)
        heading_layout.addWidget(QLabel("Adjust Heading:"))
        heading_layout.addWidget(self.heading_input)
        heading_layout.addWidget(self.set_heading_btn)
        vbox.addLayout(heading_layout)

        # Layout cho thanh trượt điều khiển tốc độ (PWM 0-255)
        speed_layout = QHBoxLayout()
        self.speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 255)
        self.speed_slider.setValue(128)  # Giá trị mặc định
        self.speed_value_label = QLabel("128")
        self.speed_slider.valueChanged.connect(self.change_speed)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        vbox.addLayout(speed_layout)

        # Layout chứa các nút điều khiển di chuyển và nhãn trạng thái
        btn_layout = QHBoxLayout()
        self.forward_btn = QPushButton("Tiến")
        self.forward_btn.pressed.connect(lambda: self.lidar.send_command("forward"))
        self.forward_btn.released.connect(lambda: self.lidar.send_command("stop"))

        self.backward_btn = QPushButton("Lùi")
        self.backward_btn.pressed.connect(lambda: self.lidar.send_command("reverse"))
        self.backward_btn.released.connect(lambda: self.lidar.send_command("stop"))

        # Nút xoay trái
        self.left_btn = QPushButton("Xoay trái")
        self.left_btn.pressed.connect(lambda: self.lidar.send_command("left"))
        self.left_btn.released.connect(lambda: self.lidar.send_command("stop"))

        # Nút xoay phải
        self.right_btn = QPushButton("Xoay phải")
        self.right_btn.pressed.connect(lambda: self.lidar.send_command("right"))
        self.right_btn.released.connect(lambda: self.lidar.send_command("stop"))

        # Nhãn hiển thị quãng đường đã đi (cm)
        self.status_label = QLabel("Robot Distance: 0.00 cm")
        btn_layout.addWidget(self.forward_btn)
        btn_layout.addWidget(self.backward_btn)
        btn_layout.addWidget(self.left_btn)
        btn_layout.addWidget(self.right_btn)
        btn_layout.addWidget(self.status_label)
        vbox.addLayout(btn_layout)

        # QTimer để cập nhật giao diện định kỳ (mỗi 100ms)
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

    def change_speed(self, value):
        self.speed_value_label.setText(str(value))
        # Gửi lệnh cập nhật tốc độ đến ESP32, ví dụ: "set_speed 128"
        command = f"set_speed {value}"
        self.lidar.send_command(command)

    def reconnect_device(self):
        new_ip = self.ip_input.text().strip()
        if new_ip:
            success = self.lidar.reconnect(new_ip)
            if success:
                self.status_label.setText(f"Kết nối thành công tới {new_ip}")
            else:
                self.status_label.setText("Kết nối thất bại. Kiểm tra lại địa chỉ IP.")

    def update_gui(self):
        # Cập nhật bản đồ nếu có dữ liệu mới
        if not self.lidar.plot_queue.empty():
            while not self.lidar.plot_queue.empty():
                self.lidar.plot_queue.get()
            self.lidar._plot_map()
            self.canvas.draw()
        # Cập nhật nhãn trạng thái với quãng đường đi
        self.status_label.setText(f"Robot Distance: {self.lidar.robot_distance:.2f} cm")

    # ---------------------------
    # Xử lý sự kiện bàn phím để điều khiển xe
    # ---------------------------
    def keyPressEvent(self, event):
        # Bỏ qua các sự kiện lặp tự động
        if event.isAutoRepeat():
            return super().keyPressEvent(event)

        key = event.key()
        if key == Qt.Key_Up:
            self.lidar.send_command("forward")
        elif key == Qt.Key_Down:
            self.lidar.send_command("reverse")
        elif key == Qt.Key_Left:
            self.lidar.send_command("left")
        elif key == Qt.Key_Right:
            self.lidar.send_command("right")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # Bỏ qua các sự kiện lặp tự động
        if event.isAutoRepeat():
            return super().keyReleaseEvent(event)

        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self.lidar.send_command("stop")
        else:
            super().keyReleaseEvent(event)

    def closeEvent(self, event):
        self.lidar.cleanup()
        event.accept()

    def on_set_heading(self):
        # 1) Đọc giá trị góc từ input
        try:
            theta_deg = float(self.heading_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Θ phải là số (độ) hợp lệ!")
            return

        # 2) Chuyển sang radian
        theta = theta_deg * pi / 180

        # 3) Gọi backend và bắt lỗi
        try:
            self.lidar.set_heading(theta)
        except Exception as e:
            QMessageBox.critical(self, "Error Setting Heading",
                                 f"Không thể điều chỉnh hướng:\n{e}")
            return

        # 4) Cập nhật status
        self.status_label.setText(f"Hướng đã chỉnh: θ={theta_deg:.1f}°")


