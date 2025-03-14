#include <WiFi.h>
#include <WebServer.h>
#include <EEPROM.h>
#include <HardwareSerial.h>

// WiFi credentials mặc định
char ssid[32] = "DIEN";
char password[32] = "diennguyen123";
WiFiServer wifiServer(80);  // Server cho điều khiển động cơ
WebServer webServer(8080);  // Server cho cấu hình WiFi

// Serial communication packet specification
#define ANGLE_IDX 1
#define SPEED_LSB 2
#define SPEED_MSB 3
#define DATA_1 4
#define DATA_2 8
#define DATA_3 12
#define DATA_4 16 
#define CHECKSUM_LSB 20
#define CHECKSUM_MSB 21
#define PACKET_SIZE 22  
#define DATA_SIZE 7 
#define BUFFER_SIZE 50

#define RX_PIN 16
#define MOTOR_PIN 5
#define BAUDRATE_SENSOR 115200  
#define MAX_POWER 255   
#define MOTOR_SPEED 250

#define MOTOR_IN1 18
#define MOTOR_IN2 19
#define ENA 21
#define ENCODER_PIN 22

// Thêm chân cho Motor 2
#define MOTOR_IN3 23
#define MOTOR_IN4 25
#define ENB 26
#define ENCODER_PIN2 27

// Biến encoder cho bánh xe thứ hai
volatile long encoderCount2 = 0; // Đếm xung encoder bánh xe thứ hai
int motorDirection2 = 0;         // Hướng motor thứ hai (1: tiến, -1: lùi, 0: dừng)

const int encoderMin = -32768;
const int encoderMax = 32767;

// Định nghĩa cho EEPROM
#define EEPROM_SIZE 128
#define SSID_ADDR 0
#define PASS_ADDR 32

// Biến toàn cục
volatile long encoderCount = 0;
int motorDirection = 0;
int data[DATA_SIZE]; 
uint8_t packet[PACKET_SIZE];    
uint8_t lidarBuffer[PACKET_SIZE * BUFFER_SIZE];
int bufferIndex = 0;
const unsigned char HEAD_BYTE = 0xFA;
unsigned int packetIndex = 0;
bool waitPacket = true;
volatile bool newCommand = false;
String pendingCommand = "";

// PID parameters
double kp = 2.0, ki = 0.3, kd = 0.3;
double proportionalTerm = 0;
double derivativeTerm = 0; 
double integralTerm = 0;
double previousSpeed = 0;
int controlEffort = 0;
unsigned long lastPIDTime = 0;
unsigned long lastSendTime = 0;

HardwareSerial lidarSerial(1);
WiFiClient client;

// Biến lưu tốc độ hiện tại được điều khiển thông qua ENA (PWM 0-255)
int currentMotorSpeed = MOTOR_SPEED;

// Interrupt Service Routine cho encoder
void IRAM_ATTR encoderISR() {
    if (motorDirection == 1) {
        if (encoderCount < encoderMax) encoderCount++;
        else encoderCount = encoderMin;
    } else if (motorDirection == -1) {
        if (encoderCount > encoderMin) encoderCount--;
        else encoderCount = encoderMax;
    }
}

void IRAM_ATTR encoderISR2() {
    if (motorDirection2 == 1) { // Tiến
        if (encoderCount2 < encoderMax) encoderCount2++;
        else encoderCount2 = encoderMin;
    } else if (motorDirection2 == -1) { // Lùi
        if (encoderCount2 > encoderMin) encoderCount2--;
        else encoderCount2 = encoderMax;
    }
}

// Đọc dữ liệu từ EEPROM
void readEEPROM() {
    for (int i = 0; i < 32; i++) {
        ssid[i] = EEPROM.read(SSID_ADDR + i);
        password[i] = EEPROM.read(PASS_ADDR + i);
    }
}

// Ghi dữ liệu vào EEPROM
void writeEEPROM(const char* newSSID, const char* newPass) {
    for (int i = 0; i < 32; i++) {
        EEPROM.write(SSID_ADDR + i, i < strlen(newSSID) ? newSSID[i] : 0);
        EEPROM.write(PASS_ADDR + i, i < strlen(newPass) ? newPass[i] : 0);
    }
    EEPROM.commit();
}

// Trang HTML để cấu hình WiFi
String htmlPage = "<!DOCTYPE html><html><body><h2>WiFi Config</h2>"
                  "<form action=\"/save\" method=\"POST\">"
                  "SSID: <input type=\"text\" name=\"ssid\"><br>"
                  "Password: <input type=\"text\" name=\"pass\"><br>"
                  "<input type=\"submit\" value=\"Save\"></form>"
                  "</body></html>";

// Kết nối WiFi
bool connectWiFi() {
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    return WiFi.status() == WL_CONNECTED;
}

// Khởi động chế độ AP
void startAPMode() {
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32_Config", "12345678");
    Serial.println("AP Mode started. IP: ");
    Serial.println(WiFi.softAPIP());
}

// Xử lý yêu cầu gốc cho WebServer
void handleRoot() {
    webServer.send(200, "text/html", htmlPage);
}

// Xử lý lưu WiFi mới
void handleSave() {
    String newSSID = webServer.arg("ssid");
    String newPass = webServer.arg("pass");
    newSSID.trim();
    newPass.trim();
    writeEEPROM(newSSID.c_str(), newPass.c_str());
    strcpy(ssid, newSSID.c_str());
    strcpy(password, newPass.c_str());
    webServer.send(200, "text/html", "<h2>WiFi Saved! Rebooting...</h2>");
    delay(1000);
    ESP.restart();
}

void setup() {
    Serial.begin(115200);
    EEPROM.begin(EEPROM_SIZE);
    readEEPROM();

    WiFi.begin(ssid, password);
    pinMode(MOTOR_PIN, OUTPUT);
    lidarSerial.begin(BAUDRATE_SENSOR, SERIAL_8N1, RX_PIN, -1);
    analogWrite(MOTOR_PIN, 125);
    lastPIDTime = millis();

    // Cấu hình motor 1
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(ENA, OUTPUT);
    pinMode(ENCODER_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), encoderISR, CHANGE);

    // Cấu hình motor 2
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);
    pinMode(ENB, OUTPUT);
    pinMode(ENCODER_PIN2, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN2), encoderISR2, CHANGE);

    motorStopBoth(); // Dừng cả hai motor khi khởi động

    // Kết nối WiFi hoặc chuyển sang AP (giữ nguyên phần còn lại)
    if (!connectWiFi()) {
        Serial.println("\nWiFi connection failed. Starting AP mode...");
        startAPMode();
        webServer.on("/", handleRoot);
        webServer.on("/save", handleSave);
        webServer.begin();
    } else {
        Serial.println("\nWiFi connected");
        Serial.println(WiFi.localIP());
        wifiServer.begin();
        webServer.on("/", handleRoot);
        webServer.on("/save", handleSave);
        webServer.begin();
    }
    Serial.println("Encoder monitoring started");

}

void loop() {
    webServer.handleClient(); // Xử lý yêu cầu cấu hình WiFi

    handleClientCommands();
    if (!newCommand) {
        handleLidarData();
    }

    unsigned long currentTime = millis();
    if (currentTime - lastPIDTime >= 100) {
        motorSpeedPID(MOTOR_SPEED, data[1]);
        lastPIDTime = currentTime;
    }

    if (currentTime - lastSendTime >= 20) {
        if (client.connected() && bufferIndex > 0) {
            sendDataToClient();
            lastSendTime = currentTime;
            bufferIndex = 0;
        }
    }
}

void handleClientCommands() {
    if (!client || !client.connected()) {
        client = wifiServer.available();
        if (client) {
            Serial.println("Client connected");
        }
    }
    if (client.connected() && client.available()) {
        String command = client.readStringUntil('\n');
        command.trim();
        Serial.println("Received command: " + command);
        newCommand = true;
        pendingCommand = command;
        
        if (pendingCommand == "forward") {
            motorForwardBoth();
        } else if (pendingCommand == "reverse") {
            motorReverseBoth();
        } else if (pendingCommand == "left") {
            motorTurnLeft();
        } else if (pendingCommand == "right") {
            motorTurnRight();
        } else if (pendingCommand == "stop") {
            motorStopBoth();
        } else if (pendingCommand.startsWith("set_speed")) {
            int speedVal = pendingCommand.substring(9).toInt();
            speedVal = constrain(speedVal, 0, 255);
            currentMotorSpeed = speedVal;
            // Cập nhật tốc độ cho cả hai motor nếu đang chạy
            if (motorDirection != 0 || motorDirection2 != 0) {
                analogWrite(ENA, currentMotorSpeed);
                analogWrite(ENB, currentMotorSpeed);
            }
            Serial.print("Motor speed set to: ");
            Serial.println(currentMotorSpeed);
        }
        newCommand = false;
    }
}

void handleLidarData() {
    while (lidarSerial.available() > 0 && bufferIndex < BUFFER_SIZE * PACKET_SIZE) {
        uint8_t receivedByte = lidarSerial.read();
        if (waitPacket && receivedByte == HEAD_BYTE) {
            packetIndex = 0;
            waitPacket = false;
            packet[packetIndex++] = receivedByte;
        } else if (!waitPacket) {
            packet[packetIndex++] = receivedByte;
            if (packetIndex >= PACKET_SIZE) {
                waitPacket = true;
                decodePacket(packet, PACKET_SIZE);
                memcpy(lidarBuffer + bufferIndex, packet, PACKET_SIZE);
                bufferIndex += PACKET_SIZE;
            }
        }
    }
}

void motorForwardBoth() {
    // Motor 1 tiến
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(ENA, currentMotorSpeed);
    motorDirection = 1;
    
    // Motor 2 tiến
    digitalWrite(MOTOR_IN3, HIGH);
    digitalWrite(MOTOR_IN4, LOW);
    analogWrite(ENB, currentMotorSpeed);
    motorDirection2 = 1;
    Serial.println("Motors: Forward");
}

void motorReverseBoth() {
    // Motor 1 lùi
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, HIGH);
    analogWrite(ENA, currentMotorSpeed);
    motorDirection = -1;
    
    // Motor 2 lùi
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, HIGH);
    analogWrite(ENB, currentMotorSpeed);
    motorDirection2 = -1;
    Serial.println("Motors: Reverse");
}

void motorTurnLeft() {
    // Motor 1 lùi
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, HIGH);
    analogWrite(ENA, currentMotorSpeed);
    motorDirection = -1;
    
    // Motor 2 tiến
    digitalWrite(MOTOR_IN3, HIGH);
    digitalWrite(MOTOR_IN4, LOW);
    analogWrite(ENB, currentMotorSpeed);
    motorDirection2 = 1;
    Serial.println("Motors: Turn Left");
}

void motorTurnRight() {
    // Motor 1 tiến
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(ENA, currentMotorSpeed);
    motorDirection = 1;
    
    // Motor 2 lùi
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, HIGH);
    analogWrite(ENB, currentMotorSpeed);
    motorDirection2 = -1;
    Serial.println("Motors: Turn Right");
}

void motorStopBoth() {
    // Dừng motor 1
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(ENA, 0);
    motorDirection = 0;
    
    // Dừng motor 2
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, LOW);
    analogWrite(ENB, 0);
    motorDirection2 = 0;
    Serial.println("Motors: Stopped");
}

void decodePacket(uint8_t packet[], int packetSize) {
    int data_idx = 0;
    for (int idx = 0; idx < DATA_SIZE; idx++) data[idx] = 0;
    for (int i = 0; i < packetSize; i++) {
        if (i == ANGLE_IDX) {
            int angle = (packet[i] - 0xA0) * 4;
            if (angle > 360) return;
            data[data_idx++] = angle;
        } else if (i == SPEED_LSB) {
            int speed = ((packet[SPEED_MSB] << 8) | packet[SPEED_LSB]) / 64;
            data[data_idx++] = speed;
        } else if (i == DATA_1 || i == DATA_2 || i == DATA_3 || i == DATA_4) {
            uint16_t distance = ((packet[i+1] & 0x3F) << 8) | packet[i];
            data[data_idx++] = distance;
        }
    }
    data[data_idx] = checksum(packet, PACKET_SIZE - 2);
}

void sendDataToClient() {
    String dataString = "";
    for (int i = 0; i < DATA_SIZE; i++) {
        dataString += String(data[i]) + "\t";
    }
    dataString += String(encoderCount) + "\t" + String(encoderCount2); // Gửi cả hai encoder
    client.println(dataString);
    Serial.println("Sent: " + dataString);
}

uint16_t checksum(uint8_t packet[], uint8_t size) {
    uint32_t chk32 = 0;
    for (int i = 0; i < size / 2; i++) {
        chk32 = (chk32 << 1) + ((packet[i * 2 + 1] << 8) + packet[i * 2]);
    }
    return (uint16_t)((chk32 & 0x7FFF) + (chk32 >> 15)) & 0x7FFF;
}

void motorSpeedPID(int targetSpeed, int currentSpeed) {
    unsigned long currentTime = millis();
    double deltaT = (currentTime - lastPIDTime) / 1000.0;
    if (deltaT <= 0) deltaT = 0.1;

    proportionalTerm = targetSpeed - currentSpeed;
    derivativeTerm = (currentSpeed - previousSpeed) / deltaT;
    integralTerm += proportionalTerm * deltaT;
    integralTerm = constrain(integralTerm, -50, 50);

    controlEffort = kp * proportionalTerm + kd * derivativeTerm + ki * integralTerm;
    controlEffort = constrain(controlEffort, 0, MAX_POWER);
    analogWrite(MOTOR_PIN, controlEffort);
    
    previousSpeed = currentSpeed;
}
