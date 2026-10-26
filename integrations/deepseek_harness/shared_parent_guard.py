"""Shared container-bound execution repair; never reads benchmark answers."""
from __future__ import annotations

import hashlib
import json
import re


class SharedParentGuard:
    def __init__(self, host_cwd: str):
        self.host_cwd = host_cwd
        self.container_cwd = None
        self.last_action = None
        self.last_receipt = None
        self.unchanged = 0

    def resolve(self, arguments: dict) -> tuple[dict, str | None]:
        resolved = dict(arguments)
        cwd = resolved.get('cwd')
        if cwd and (cwd == self.host_cwd or cwd.startswith(self.host_cwd + '/')):
            if self.container_cwd is None:
                return resolved, 'Container cwd discovery is unavailable; replan.'
            resolved['cwd'] = self.container_cwd.rstrip('/') + cwd[len(self.host_cwd):]
        # Never rewrite a shell program or file contents by string substitution.
        # Invalid host references request a new literal native command instead.
        if any(re.search(r'/(?:mnt/phiagent-data|data0/phiagent|opt/phiagent/work|Users)/', str(v))
               for v in (resolved.get('command', ''), resolved.get('cwd', ''))):
            return resolved, ('Host path is outside the benchmark container. '
                              f'Container initial cwd is {self.container_cwd!r}. '
                              'Use container paths from public task instructions or pwd; replan.')
        key = (resolved['command'], resolved.get('cwd', self.container_cwd))
        command = resolved['command']
        # Blank receipts do not prove that repeated writes left filesystem state unchanged.
        readonly = bool(re.match(r'^\s*(?:pwd|ls|cat|head|tail|stat|wc|file|rg|grep)\b', command))
        readonly = readonly and not any(c in command for c in [';', '|', '>', '<', '&', '`', '$'])
        if readonly and key == self.last_action and self.unchanged >= 1:
            return resolved, ('Repeated identical command produced an unchanged public receipt. '
                              'Execution suppressed. Change the command or finish; replan from that receipt.')
        return resolved, None

    def observe(self, arguments: dict, observation: dict) -> None:
        key = (arguments['command'], arguments.get('cwd', self.container_cwd))
        receipt = hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()
        self.unchanged = self.unchanged + 1 if key == self.last_action and receipt == self.last_receipt else 0
        self.last_action, self.last_receipt = key, receipt
