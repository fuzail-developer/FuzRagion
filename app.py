import os
import socket
import uvicorn
from main import app


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def _find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
    for p in range(start_port, start_port + max_tries):
        if not _is_port_in_use(host, p):
            return p
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_tries - 1}")


def get_local_ip() -> str:
    """
    Prefer stable LAN IPs for mobile testing.
    Priority: 192.168.x.x > 172.16-31.x.x > 10.x.x.x > loopback.
    """
    candidates: set[str] = set()

    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                candidates.add(ip)
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            candidates.add(ip)
    except Exception:
        pass

    def rank(ip: str) -> int:
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("172."):
            parts = ip.split(".")
            if len(parts) > 1:
                try:
                    second = int(parts[1])
                    if 16 <= second <= 31:
                        return 1
                except ValueError:
                    pass
        if ip.startswith("10."):
            return 2
        return 9

    if not candidates:
        return "127.0.0.1"
    return sorted(candidates, key=rank)[0]


def _has_root_get_route() -> bool:
    for route in app.routes:
        if getattr(route, "path", None) == "/" and "GET" in getattr(route, "methods", set()):
            return True
    return False


if not _has_root_get_route():
    @app.get("/")
    def root():
        return {"status": "running"}


if __name__ == "__main__":
    host = "0.0.0.0"
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
    else:
        base_port = int(os.environ.get("API_PORT", "8001"))
        port = base_port
        if _is_port_in_use(host, port):
            port = _find_available_port(host, port + 1)

    local_ip = get_local_ip()

    print("\nServer Started\n")
    print(f"Use on Laptop: http://127.0.0.1:{port}/app")
    print(f"Use on Mobile: http://{local_ip}:{port}/app")
    print("Note: mobile and laptop must be on same Wi-Fi.\n")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
