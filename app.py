import os
import socket
from pathlib import Path

import uvicorn


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


def main() -> int:
    requested_port = int(os.environ.get("PORT", 8001))
    is_hosted = any(
        os.getenv(name)
        for name in ("RENDER", "RENDER_SERVICE_ID", "RENDER_EXTERNAL_URL", "RENDER_EXTERNAL_HOSTNAME")
    )
    port = requested_port
    if not is_hosted:
        while not port_is_free(port) and port < requested_port + 20:
            port += 1

    ip = get_ip()
    base_dir = Path(__file__).resolve().parent
    public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv(
        "RENDER_EXTERNAL_HOSTNAME", ""
    )

    print("\n" + "=" * 90)
    print("Fuzragion Server Started Successfully!")
    print("=" * 90)
    print(f"PID      : {os.getpid()}")
    print(f"Project  : {base_dir}")
    if is_hosted:
        print("Mode     : Render hosted deployment")
        print(f"Bind     : 0.0.0.0:{port}")
        if public_url:
            print(f"Public   : {public_url}")
    elif port != requested_port:
        print(f"Port {requested_port} was busy, switched to {port}")
    print(f"Frontend : http://127.0.0.1:{port}/")
    print(f"App Route : http://127.0.0.1:{port}/app")
    print(f"Network  : http://{ip}:{port}/")
    if "OneDrive" in str(base_dir):
        print("Note     : Project is inside OneDrive; sync/lock issues can interrupt local services.")
    print("=" * 90 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
        access_log=True,
        reload=False,
        factory=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
