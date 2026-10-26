#!/usr/bin/env python3
"""Serve hash-pinned official installer artifacts on the private Docker bridge."""

from __future__ import annotations

import argparse
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import shutil
import socket
from urllib.parse import quote, unquote, urlsplit

try:
    from .terminal_docker_network import docker_client_allowed
except ImportError:
    from terminal_docker_network import docker_client_allowed


ARTIFACTS = {
    '/github/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz': {
        'filename': 'uv-x86_64-unknown-linux-gnu.tar.gz',
        'size': 21370871,
        'sha256': '2cf10babba653310606f8b49876cfb679928669e7ddaa1fb41fb00ce73e64f66',
    },
    '/github/astral-sh/python-build-standalone/releases/download/20251014/cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz': {
        'filename': 'cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz',
        'size': 33517986,
        'sha256': 'c0ccda275948c79996e46993c2c5476ff5cca606dee530f1dea712179131b348',
    },
}


def checked_artifacts(directory, records=ARTIFACTS):
    checked = {}
    for route, row in records.items():
        target = directory / row['filename']
        if target.is_symlink() or not target.is_file() or target.stat().st_size != row['size']:
            raise ValueError('official artifact size or path mismatch: ' + row['filename'])
        value = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(chunk)
        if value.hexdigest() != row['sha256']:
            raise ValueError('official artifact checksum mismatch: ' + row['filename'])
        checked[route] = {**row, 'path': str(target), 'verified_stat': (target.stat().st_size, target.stat().st_mtime_ns)}
    return checked


def checked_wheels(root):
    manifest = root / 'wheels-manifest.json'
    if not manifest.is_file():
        return {}, b''
    entries = json.loads(manifest.read_text())['files']
    records = {}
    links = []
    for row in entries:
        name = row['filename']
        origin = urlsplit(row['official_url'])
        if (Path(name).name != name or not name.endswith('.whl') or origin.scheme != 'https'
                or origin.hostname != 'files.pythonhosted.org' or origin.username or origin.password):
            raise ValueError('wheel must retain its exact official PyPI artifact origin')
        route = '/wheels/' + name
        if route in records:
            raise ValueError('duplicate wheel identity')
        records[route] = {'filename': name, 'size': row['size'], 'sha256': row['sha256'],
                          'official_url': row['official_url']}
        links.append('<a href="' + quote(name) + '#sha256=' + row['sha256'] + '">' + html.escape(name) + '</a>')
    checked = checked_artifacts(root / 'wheels', records)
    return checked, ('<!doctype html><html><body>\n' + '\n'.join(links) + '\n</body></html>').encode()


def resolve_route(raw_path, records):
    parts = urlsplit(raw_path)
    route = unquote(parts.path)
    if parts.query or parts.fragment or parts.netloc or '..' in Path(route).parts:
        raise ValueError('invalid artifact route')
    if route in records:
        return records[route], None
    allowed = ('/github/astral-sh/uv/releases/download/',
               '/github/astral-sh/python-build-standalone/releases/download/')
    if not route.startswith(allowed):
        raise ValueError('artifact is outside official installer namespaces')
    return None, 'https://github.com/' + quote(route.removeprefix('/github/'), safe='/+')


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_artifact(head_only=True)

    def do_GET(self):
        self.send_artifact(head_only=False)

    def send_artifact(self, *, head_only):
        if not docker_client_allowed(self.client_address[0]):
            self.send_error(403)
            return
        if self.path == '/wheels/' and self.server.wheel_index:
            payload = self.server.wheel_index
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)
            return
        try:
            row, redirect = resolve_route(self.path, self.server.records)
        except ValueError:
            self.send_error(404)
            return
        if redirect:
            self.send_response(302)
            self.send_header('Location', redirect)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        target = Path(row['path'])
        if target.is_symlink() or (target.stat().st_size, target.stat().st_mtime_ns) != row['verified_stat']:
            self.send_error(503, 'cached artifact changed')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(row['size']))
        self.send_header('ETag', '"sha256:' + row['sha256'] + '"')
        self.end_headers()
        if not head_only:
            with target.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18996)
    args = parser.parse_args()
    if socket.gethostname() != 'harnessmm-api-worker':
        parser.error('official dependency cache is restricted to A800-4')
    if not 1024 <= args.port <= 65535:
        parser.error('dependency cache port must be between 1024 and 65535')
    root = args.root.resolve()
    records = checked_artifacts(root / 'artifacts')
    wheels, wheel_index = checked_wheels(root)
    records.update(wheels)
    with ThreadingHTTPServer(('172.17.0.1', args.port), Handler) as server:
        server.records = records
        server.wheel_index = wheel_index
        receipt = {'status': 'READY_HASH_VERIFIED_OFFICIAL_ARTIFACTS',
                   'bind': f'172.17.0.1:{args.port}',
                   'artifacts': records, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'client_policy_sha256': hashlib.sha256(Path(docker_client_allowed.__code__.co_filename).read_bytes()).hexdigest(),
                   'client_policy': '172.16/12 plus verified local Docker bridge subnets in 192.168/16',
                   'unknown_versions': 'redirect to exact original official namespace, never substitute'}
        with (root / 'READY.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
        server.serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
