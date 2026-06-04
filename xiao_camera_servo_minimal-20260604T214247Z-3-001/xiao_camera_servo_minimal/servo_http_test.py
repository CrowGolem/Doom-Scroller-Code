import requests

BASE_URL = "http://192.168.4.1"
TIMEOUT = 3

COMMANDS = {
    "1": ("forward", "Scroll forward"),
    "2": ("backward", "Scroll backward"),
    "3": ("doubletap", "Double tap"),
    "h": ("home", "Home"),
    "s": ("status", "Status"),
}

def send(params):
    try:
        r = requests.get(f"{BASE_URL}/servo", params=params, timeout=TIMEOUT)
        print(f"{r.url} -> {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print("Request failed:", repr(e))

print("Camera HTTP servo test")
print(f"Base URL: {BASE_URL}")
print("Commands: 1=forward, 2=backward, 3=doubletap, h=home, s=status")
print("Direct angle: a1 25, a1 120, a2 90, a2 120")
print("q=quit")

while True:
    command = input("command> ").strip().lower()
    if command == "q":
        break
    if command in COMMANDS:
        cmd, label = COMMANDS[command]
        print(label)
        send({"cmd": cmd})
        continue
    if command.startswith("a1 ") or command.startswith("a2 "):
        parts = command.split()
        if len(parts) != 2 or not parts[1].isdigit():
            print("Use: a1 90 or a2 120")
            continue
        servo = 1 if parts[0] == "a1" else 2
        angle = int(parts[1])
        print(f"Set servo {servo} to {angle}")
        send({"cmd": "set", "s": servo, "a": angle})
        continue
    print("Invalid command")
