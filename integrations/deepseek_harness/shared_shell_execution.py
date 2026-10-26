"""Common shell status and bounded process cleanup, with literal args preserved."""
import shlex

def wrapped_command(command, timeout_sec, token):
    assert token.isalnum()
    marker='/tmp/.phiagent-terminal-'+token+'.pid'
    body='read -r p comm state ppid pgid rest < /proc/"$BASHPID"/stat; printf \'%s\\n\' "$pgid" > '+shlex.quote(marker)+'; exec bash -o pipefail -lc '+shlex.quote(command)
    if timeout_sec is not None:
        assert type(timeout_sec) is int and timeout_sec>0
        wrapper='timeout --signal=TERM --kill-after=5 '+str(timeout_sec)+' bash -c '+shlex.quote(body)
    else:
        wrapper='setsid bash -c '+shlex.quote(body)
    return wrapper, marker

def cleanup_command(marker):
    return ('if test -f '+shlex.quote(marker)+'; then p=$(cat '+shlex.quote(marker)+'); '
            'case "$p" in ""|*[!0-9]*) exit 2;; esac; kill -TERM -- -"$p" 2>/dev/null || true; fi')
