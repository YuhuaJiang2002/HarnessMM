"""Admit local Docker bridge clients without exposing dependency proxies publicly."""

from __future__ import annotations

import ipaddress
import json
import subprocess
import threading


_STANDARD = ipaddress.ip_network("172.16.0.0/12")
_DOCKER_FALLBACK = ipaddress.ip_network("192.168.0.0/16")
_networks = ()
_lock = threading.Lock()


def docker_client_allowed(address):
    global _networks
    try:
        client = ipaddress.ip_address(address)
    except ValueError:
        return False
    if client in _STANDARD:
        return True
    if client not in _DOCKER_FALLBACK:
        return False
    with _lock:
        if any(client in network for network in _networks):
            return True
        try:
            identifiers = subprocess.check_output(
                ["docker", "network", "ls", "--filter", "driver=bridge", "--format", "{{.ID}}"],
                text=True, timeout=10).split()
            if not identifiers:
                return False
            inventory = json.loads(subprocess.check_output(
                ["docker", "network", "inspect", *identifiers], text=True, timeout=10))
            discovered = []
            for record in inventory:
                if record.get("Driver") != "bridge" or record.get("Scope") != "local":
                    continue
                for config in record.get("IPAM", {}).get("Config", []) or []:
                    subnet = ipaddress.ip_network(config["Subnet"])
                    if subnet.version == 4 and subnet.subnet_of(_DOCKER_FALLBACK):
                        discovered.append(subnet)
            _networks = tuple(discovered)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
            return False
        return any(client in network for network in _networks)
