#!/usr/bin/env python3
"""Cache official APT downloads on the private evaluation Docker network."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import shutil
import socket
import threading
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

try:
    from .terminal_docker_network import docker_client_allowed
except ImportError:
    from terminal_docker_network import docker_client_allowed


ORIGINS = {
    'deb.debian.org': ('/debian/', '/debian-security/'),
    'security.debian.org': ('/debian-security/', '/debian/'),
    'archive.ubuntu.com': ('/ubuntu/',),
    'security.ubuntu.com': ('/ubuntu/',),
}
CHUNK_BYTES = 131072
MAX_BYTES = 256 * 1024 * 1024


def official_origin(raw_url):
    parts = urlsplit(raw_url)
    if (parts.scheme != 'http' or parts.hostname not in ORIGINS or parts.port not in (None, 80)
            or parts.username or parts.password or parts.query or parts.fragment
            or '..' in Path(unquote(parts.path)).parts
            or not parts.path.startswith(ORIGINS[parts.hostname])):
        raise ValueError('only official APT HTTP download URLs are admitted')
    return urlunsplit(('https', parts.hostname, parts.path, '', ''))


def checked_range(headers, payload, start, end, size):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('Content-Range', ''))
    if not match or tuple(map(int, match.groups())) != (start, end, size) or len(payload) != end - start + 1:
        raise ValueError('official APT range response differs from requested bytes')


def public_http_download(raw_url):
    parts = urlsplit(raw_url)
    if (parts.scheme != 'http' or not parts.hostname or parts.port not in (None, 80)
            or parts.username or parts.password or parts.fragment):
        raise ValueError('invalid public HTTP download')
    addresses = socket.getaddrinfo(parts.hostname, 80, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError('HTTP forwarding cannot access private or reserved addresses')
    return raw_url


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class AptCache:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.opener = build_opener(ProxyHandler({}))
        self.lock = threading.Lock()
        self.downloads = {}
        self.pool = ThreadPoolExecutor(max_workers=8)

    def request(self, url, *, method='GET', headers=None):
        try:
            return self.opener.open(Request(url, method=method, headers=headers or {}), timeout=60)
        except HTTPError as error:
            code = error.code
            error.close()
            raise OSError(f'official APT upstream HTTP {code}: {url}') from None

    def get(self, raw_url):
        origin = official_origin(raw_url)
        key = hashlib.sha256(origin.encode()).hexdigest()
        with self.lock:
            guard = self.downloads.setdefault(key, threading.Lock())
        with guard:
            directory = self.root / key
            directory.mkdir(exist_ok=True)
            payload_path = directory / 'payload'
            receipt_path = directory / 'RECEIPT.json'
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                immutable = '/by-hash/' in origin or '/pool/' in origin
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(receipt['at_utc'])).total_seconds()
                if immutable or age < 3600:
                    with payload_path.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if payload_path.stat().st_size != receipt['size'] or digest != receipt['sha256']:
                        raise ValueError('cached APT bytes changed')
                    return payload_path, receipt
            attempt = directory / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            attempt.mkdir()
            with self.request(origin, method='HEAD') as response:
                if urlsplit(response.geturl()).hostname not in ORIGINS:
                    raise ValueError('APT upstream redirected outside official origins')
                size = int(response.headers['Content-Length'])
                etag = response.headers.get('ETag')
                modified = response.headers.get('Last-Modified')
            if not 0 < size <= MAX_BYTES:
                raise ValueError('official APT download exceeds bounded cache size')

            def download(bounds):
                start, end = bounds
                headers = {'Range': f'bytes={start}-{end}'}
                if etag and not etag.startswith('W/'):
                    headers['If-Match'] = etag
                with self.request(origin, headers=headers) as response:
                    payload = response.read(end - start + 2)
                    if response.status == 200 and start == 0 and end == size - 1:
                        if len(payload) != size:
                            raise ValueError('APT response body is incomplete')
                    else:
                        checked_range(response.headers, payload, start, end, size)
                target = attempt / f'{start:012d}.part'
                target.write_bytes(payload)
                return target

            bounds = [(start, min(size - 1, start + CHUNK_BYTES - 1)) for start in range(0, size, CHUNK_BYTES)]
            parts = list(self.pool.map(download, bounds))
            pending = attempt / 'payload'
            with pending.open('xb') as output:
                for part in parts:
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, output)
            with pending.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if '/by-hash/SHA256/' in origin and origin.rsplit('/', 1)[1] != digest:
                raise ValueError('official APT by-hash content mismatch')
            receipt = {'at_utc': datetime.now(timezone.utc).isoformat(), 'original_url': raw_url,
                       'official_https_url': origin, 'size': size, 'sha256': digest,
                       'etag': etag, 'last_modified': modified, 'apt_signature_checks_disabled': False}
            (attempt / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
            pending.replace(payload_path)
            shutil.copy2(attempt / 'RECEIPT.json', directory / 'receipt.pending')
            (directory / 'receipt.pending').replace(receipt_path)
            return payload_path, receipt


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.serve_file(head_only=False)

    def do_HEAD(self):
        self.serve_file(head_only=True)

    def serve_file(self, *, head_only):
        if not docker_client_allowed(self.client_address[0]):
            self.send_error(403)
            return
        try:
            if urlsplit(self.path).hostname not in ORIGINS:
                self.forward_public(head_only=head_only)
                return
            target, receipt = self.server.cache.get(self.path)
        except (OSError, ValueError, KeyError) as error:
            self.log_error('%s', error)
            try:
                self.send_error(502, 'official dependency download unavailable')
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(200)
        self.send_header('Content-Length', str(receipt['size']))
        if receipt.get('last_modified'):
            self.send_header('Last-Modified', receipt['last_modified'])
        self.end_headers()
        if not head_only:
            try:
                with target.open('rb') as source:
                    shutil.copyfileobj(source, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass


    def forward_public(self, *, head_only):
        url = public_http_download(self.path)
        excluded = {'connection', 'proxy-connection', 'proxy-authorization', 'transfer-encoding',
                    'keep-alive', 'te', 'trailer', 'upgrade'}
        headers = {name: value for name, value in self.headers.items() if name.lower() not in excluded}
        request = Request(url, headers=headers, method='HEAD' if head_only else 'GET')
        try:
            response = build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=60)
        except HTTPError as error:
            response = error
        with response:
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in excluded:
                    self.send_header(name, value)
            self.end_headers()
            if not head_only:
                shutil.copyfileobj(response, self.wfile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed-root', type=Path)
    parser.add_argument('--port', type=int, default=18997)
    args = parser.parse_args()
    if socket.gethostname() != 'harnessmm-api-worker':
        parser.error('APT cache is restricted to A800-4')
    if not 1024 <= args.port <= 65535:
        parser.error('APT cache needs an unprivileged TCP port')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.seed_root:
        seed = args.seed_root.resolve()
        if not seed.is_relative_to('/path/to/data/phiagent/work/terminal-apt-cache'):
            parser.error('seed objects must belong to this campaign APT cache')
        shutil.copytree(seed / 'objects', root / 'objects')
    cache = AptCache(root / 'objects')
    try:
        with ThreadingHTTPServer(('172.17.0.1', args.port), Handler) as server:
            server.cache = cache
            receipt = {'status': 'READY_OFFICIAL_APT_CACHE', 'bind': f'172.17.0.1:{args.port}',
                       'at_utc': datetime.now(timezone.utc).isoformat(), 'origins': ORIGINS,
                       'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'client_policy_sha256': hashlib.sha256(Path(docker_client_allowed.__code__.co_filename).read_bytes()).hexdigest(),
                       'client_policy': '172.16/12 plus verified local Docker bridge subnets in 192.168/16',
                       'other_public_http_gets': 'forwarded uncached with status and redirects preserved',
                       'upstream_tls_verified': True, 'apt_signature_checks_disabled': False}
            (root / 'READY.json').write_text(json.dumps(receipt, indent=2) + '\n')
            server.serve_forever()
    finally:
        cache.pool.shutdown(wait=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
