import os
import socket

import uvicorn

from main import app


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


if __name__ == "__main__":
    requested_port = int(os.environ.get("PORT", 8001))
    port = requested_port
    while not port_is_free(port) and port < requested_port + 20:
        port += 1

    ip = get_ip()

    print("\n" + "=" * 90)
    print("Fuzragion Server Started Successfully!")
    print("=" * 90)
    if port != requested_port:
        print(f"Port {requested_port} was busy, switched to {port}")
    print(f"Frontend : http://127.0.0.1:{port}/app")
    print(f"Network  : http://{ip}:{port}/app")
    print("=" * 90 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
