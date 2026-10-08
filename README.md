# AIoT Deception & Weapon Detection System
ระบบตรวจจับพฤติกรรมพิรุธและตรวจจับอาวุธแบบเรียลไทม์ พร้อมแจ้งเตือนผ่าน MQTT (Arduino Thailand 2025)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── src/
│   ├── app.py                 # โปรแกรมหลัก (Deception + Weapon + MQTT)
│   ├── deception_detector.py  # โมดูลตรวจจับพิรุธ (Standalone)
│   └── weapon_detector.py     # โมดูลตรวจจับอาวุธ (Standalone)
├── models/
│   ├── yolov3_weapon.cfg      # Config โมเดล YOLOv3
│   └── yolov3_weapon.weights  # Weights โมเดล (โหลดแยก ดูด้านล่าง)
├── main.py                    # สคริปต์สั่งรันระบบ
├── requirements.txt           # ไลบรารีที่ต้องติดตั้ง
├── .gitignore
└── README.md
```

---

## 🚀 วิธีใช้งาน (Usage)

* **รันระบบหลัก:**
  ```bash
  python main.py
  ```
  *(กด `q` หรือ `ESC` เพื่อปิดโปรแกรม)*

* **รันแยกโมดูล:**
  ```bash
  python src/deception_detector.py  # ตรวจจับพิรุธ
  python src/weapon_detector.py     # ตรวจจับอาวุธ
  ```

---

## 📡 MQTT Topics

| Topic | ข้อมูลที่ส่ง |
| :--- | :--- |
| `/TUP/Tubpong/RISK` | ระดับความเสี่ยง (`VERY LOW` - `VERY HIGH`) |
| `/TUP/Tubpong/SCORE` | คะแนนความเสี่ยง (0 - 100) |
| `/TUP/Tubpong/ALERT_IMAGE` | ภาพเหตุการณ์ Base64 JPEG |
| `/Triampat/Tubpong/WORK` | สัญญาณควบคุม Arduino / อุปกรณ์ IoT |
