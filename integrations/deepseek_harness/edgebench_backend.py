"""SForge Docker execution with firewall rules confined to the owned work namespace."""

from __future__ import annotations

import dataclasses
import ipaddress
import io
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import tarfile
from typing import Any
from urllib.parse import urlsplit

from phiagent.data_engine.provenance import write_json_atomic


def permitted_endpoints(urls: list[str], gateway: str) -> list[tuple[str, int]]:
    ipaddress.IPv4Address(gateway)
    endpoints = set()
    for value in urls:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("Isolation accepts explicit credential-free HTTP(S) endpoints only")
        host = parsed.hostname
        if host == "host.docker.internal":
            host = gateway
        try:
            address = str(ipaddress.IPv4Address(host))
        except (ipaddress.AddressValueError, TypeError) as exc:
            raise ValueError("The scoped adapter requires literal IPv4 service endpoints") from exc
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        endpoints.add((address, port))
    if not endpoints:
        raise ValueError("Offline work needs explicitly permitted native service endpoints")
    return sorted(endpoints)


def native_endpoint_urls(endpoints: list[Any]) -> tuple[list[str], list[dict]]:
    urls, records = [], []
    for endpoint in endpoints:
        if not dataclasses.is_dataclass(endpoint):
            raise ValueError("Expected the pinned native AllowedEndpoint dataclass")
        record = dataclasses.asdict(endpoint)
        if set(record) != {"ip", "port", "hostname"}:
            raise ValueError("Native AllowedEndpoint fields changed")
        address = str(ipaddress.IPv4Address(record["ip"]))
        port = record["port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid native endpoint port")
        urls.append(f"http://{address}:{port}")
        records.append(record)
    return urls, records


def native_shell_command(command: str | list[str]) -> list[str]:
    return ["bash", "-lc", command] if isinstance(command, str) else command


def native_text_archive(data: str, filename: str) -> bytes:
    payload = (data + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(filename)
        info.size, info.mode = len(payload), 0o644
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def container_owned_archive(source: Path, filename: str, uid: int, gid: int) -> bytes:
    if any(type(value) is not int or not 0 <= value < 2**32 for value in (uid, gid)):
        raise ValueError("Container ownership must use measured numeric UID and GID")

    def ownership(info):
        info.uid, info.gid = uid, gid
        info.uname = info.gname = ""
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(source, arcname=filename, filter=ownership)
    return buffer.getvalue()


def firewall_commands(fd: int, chain: str, endpoints: list[tuple[str, int]]) -> list[list[str]]:
    if type(fd) is not int or fd < 0 or not re.fullmatch(r"PHI_EDGE_[a-f0-9]{12}", chain):
        raise ValueError("A held network-namespace descriptor and owned chain are required")
    prefix = ["nsenter", f"--net=/proc/self/fd/{fd}", "--"]
    commands = []
    for program in ("iptables", "ip6tables"):
        base = [*prefix, program, "-w", "10"]
        commands.extend([
            [*base, "-N", chain],
            [*base, "-A", chain, "-o", "lo", "-j", "ACCEPT"],
            [*base, "-A", chain, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED",
             "-j", "ACCEPT"],
        ])
        if program == "iptables":
            for address, port in endpoints:
                ipaddress.IPv4Address(address)
                if type(port) is not int or not 1 <= port <= 65535:
                    raise ValueError("Invalid allowed TCP service port")
                commands.append([
                    *base, "-A", chain, "-d", address, "-p", "tcp",
                    "--dport", str(port), "-j", "ACCEPT",
                ])
        commands.extend([
            [*base, "-A", chain, "-j", "REJECT"],
            [*base, "-I", "OUTPUT", "1", "-j", chain],
        ])
    return commands


class NamespaceIsolation:
    def __init__(
        self, container: Any, urls: list[str], gateway: str, root: Path, logger: logging.Logger,
    ):
        self.container, self.root, self.logger = container, root, logger
        self.endpoints = permitted_endpoints(urls, gateway)
        self.fd: int | None = None
        self.installed: dict[str, bool] = {}
        self.events: list[dict] = []
        self.chain = ""

    def _run(self, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        if self.fd is None:
            raise RuntimeError("No owned network namespace is held")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30,
            pass_fds=(self.fd,), check=False,
        )
        self.events.append({
            "command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        })
        write_json_atomic(self.root / "NAMESPACE_FIREWALL.json", {
            "scope": "owned work-container network namespace only",
            "host_firewall_modified": False, "events": self.events,
        })
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr,
            )
        return result

    def apply(self) -> None:
        if socket.gethostname() != "harnessmm-api-worker":
            raise RuntimeError("Scoped native execution is authorized only on A800-4")
        self.container.reload()
        attrs = self.container.attrs
        pid = attrs["State"]["Pid"]
        ident = self.container.id
        dropped = {name.removeprefix("CAP_") for name in attrs["HostConfig"].get("CapDrop") or []}
        if (
            type(pid) is not int or pid <= 0 or not re.fullmatch(r"[a-f0-9]{64}", ident)
            or attrs["HostConfig"].get("Privileged")
            or not {"NET_RAW", "NET_ADMIN"} <= dropped
        ):
            raise RuntimeError("Work container must be live, unprivileged and unable to alter networking")
        self.fd = os.open(f"/proc/{pid}/ns/net", os.O_RDONLY)
        if os.fstat(self.fd).st_ino == os.stat("/proc/self/ns/net").st_ino:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("Refusing to change the shared host network namespace")
        self.chain = "PHI_EDGE_" + ident[:12]
        try:
            for command in firewall_commands(self.fd, self.chain, self.endpoints):
                self._run(command)
                program = command[3]
                if "-N" in command:
                    self.installed[program] = False
                if "-I" in command:
                    self.installed[program] = True
            for program in ("iptables", "ip6tables"):
                self._run([
                    "nsenter", f"--net=/proc/self/fd/{self.fd}", "--", program, "-w", "10",
                    "-C", "OUTPUT", "-j", self.chain,
                ])
        except Exception:
            try:
                self.cleanup()
            except (OSError, subprocess.SubprocessError) as cleanup_error:
                self.logger.error("Owned namespace cleanup also failed: %s", cleanup_error)
            raise

    def cleanup(self) -> None:
        if self.fd is None:
            return
        failures = []
        try:
            for program, linked in self.installed.items():
                prefix = [
                    "nsenter", f"--net=/proc/self/fd/{self.fd}", "--", program, "-w", "10",
                ]
                operations = ([["-D", "OUTPUT", "-j", self.chain]] if linked else [])
                operations += [["-F", self.chain], ["-X", self.chain]]
                for operation in operations:
                    result = self._run([*prefix, *operation], check=False)
                    if result.returncode:
                        failures.append(result.stderr)
        finally:
            os.close(self.fd)
            self.fd = None
            self.installed.clear()
        if failures:
            raise RuntimeError("Owned namespace cleanup failed: " + "; ".join(failures))


class WorkContainers:
    def __init__(self, containers: Any):
        self.containers = containers

    def create(self, *args, **kwargs):
        if kwargs.get("privileged") or kwargs.get("device_requests"):
            raise ValueError("This CPU/API campaign cannot create privileged or GPU work containers")
        environment = kwargs.get("environment") or {}
        if not isinstance(environment, dict):
            raise ValueError("Native work environment must be an explicit mapping")
        kwargs["environment"] = {
            **environment, "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
        }
        kwargs["cap_drop"] = sorted(set(kwargs.get("cap_drop") or []) | {"NET_RAW", "NET_ADMIN"})
        kwargs["runtime"] = "runc"
        kwargs["labels"] = {
            **(kwargs.get("labels") or {}),
            "phiagent.campaign": "deepseek-edgebench-20260929-v1",
            "phiagent.network_scope": "owned-container-namespace",
        }
        return self.containers.create(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.containers, name)


class WorkClient:
    def __init__(self, client: Any):
        self.client = client
        self.containers = WorkContainers(client.containers)

    def __getattr__(self, name):
        return getattr(self.client, name)


def make_scoped_backend(client: Any, *, gateway: str, root: Path, relay_url: str):
    from sforge.harness.backend.docker_backend import DockerBackend

    class ScopedDockerBackend(DockerBackend):
        @property
        def backend_name(self):
            return "docker-scoped"

        def write_to_container(self, handle, data, dst):
            result = self.exec_run(handle, ["mkdir", "-p", str(dst.parent)])
            if result.exit_code:
                raise RuntimeError("Native container could not create its prompt directory")
            container = client.containers.get(handle.id)
            if not container.put_archive(str(dst.parent), native_text_archive(data, dst.name)):
                raise RuntimeError("Native container rejected the exact prompt file")

        def copy_to_container(self, handle, src, dst):
            ownership = []
            for flag in ("-u", "-g"):
                result = self.exec_run(handle, ["id", flag])
                if result.exit_code or not result.output.strip().isdecimal():
                    raise RuntimeError("Could not measure the native container user's ownership")
                ownership.append(int(result.output.strip()))
            container = client.containers.get(handle.id)
            payload = container_owned_archive(src, dst.name, *ownership)
            if not container.put_archive(str(dst.parent), payload):
                raise RuntimeError("Native container rejected the owned public framework asset")

        def exec_run(self, handle, cmd, **kwargs):
            return super().exec_run(handle, native_shell_command(cmd), **kwargs)

        def _exec_run_impl(self, handle, cmd, timeout=60, **kwargs):
            return super()._exec_run_impl(handle, native_shell_command(cmd), timeout, **kwargs)

        def create_network_isolation(self, handle, allowed_endpoints, logger):
            container = client.containers.get(handle.id)
            native_urls, native_records = native_endpoint_urls(allowed_endpoints)
            urls = [*native_urls, relay_url]
            write_json_atomic(root / "NETWORK_ENDPOINTS.json", {
                "container_id": handle.id, "allowed_endpoints": urls, "gateway": gateway,
                "native_allowed_endpoints": native_records,
            })
            return NamespaceIsolation(container, urls, gateway, root, logger)

    return ScopedDockerBackend(client=WorkClient(client))
