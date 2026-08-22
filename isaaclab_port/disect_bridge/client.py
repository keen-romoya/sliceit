"""TCP client for the DiSECt bridge server (see server.py for protocol)."""
import json
import socket


class DisectClient:
    def __init__(self, host="127.0.0.1", port=8299, timeout=60.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.rfile = self.sock.makefile("r")

    def _rpc(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())
        resp = json.loads(self.rfile.readline())
        if not resp.get("ok"):
            raise RuntimeError(f"disect bridge error: {resp.get('error')}")
        return resp

    def reset(self):
        return self._rpc({"cmd": "reset"})

    def step(self, pos, vel, substeps=50, include_mesh=False):
        return self._rpc({"cmd": "step", "pos": list(map(float, pos)),
                          "vel": list(map(float, vel)), "substeps": substeps,
                          "include_mesh": include_mesh})

    def mesh_topology(self):
        return self._rpc({"cmd": "mesh_topology"})

    def info(self):
        return self._rpc({"cmd": "info"})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
