# Test 2: Camera + minimal servo endpoint

This is your camera project with one added endpoint: `/servo`. It uses `ESP32Servo` and includes direct angle testing.

Upload `xiao_camera_servo_minimal.ino` from this folder with the other files in the same folder. Connect to Wi-Fi `XIAO-Camera` password `12345678`. Run `python servo_http_test.py`.

Important: test direct angle commands first: `a1 25`, `a1 120`, `a2 90`, `a2 120`.
