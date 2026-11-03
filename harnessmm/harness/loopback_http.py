"""Bind native loopback services on ports accepted by the SDK's Fetch client."""

from http.server import ThreadingHTTPServer

# Fetch's bad-port set includes SIP ports 5060/5061, even on loopback. Some
# authorized hosts allocate these ports for bind(0).
FETCH_BLOCKED_PORTS = frozenset((
    0, 1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53,
    69, 77, 79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117,
    119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514,
    515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989,
    990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061,
    6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080,
))


class FetchCompatibleHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler):
        if address[0] != "127.0.0.1":
            raise ValueError("Native services must remain loopback-only")
        if address[1] and address[1] in FETCH_BLOCKED_PORTS:
            raise ValueError("Explicit native port is blocked by Fetch")
        for _ in range(128):
            super().__init__(address, handler)
            if self.server_port not in FETCH_BLOCKED_PORTS:
                return
            self.server_close()
        raise OSError("No Fetch-compatible loopback port could be allocated")
