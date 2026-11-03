"""Input-independent acceptance checks; no reference answer or scorer access."""
import re


def inspect_draft(response, finish_reason):
    lines = response.strip().splitlines()
    answer = re.fullmatch(r'ANSWER:\s*([0-9]{1,3})\s*', lines[-1]) if lines else None
    confidence = re.findall(r'^CONFIDENCE:\s*(\d{1,3})\s*$', response, re.M)
    checks = {
        'complete_generation': finish_reason == 'stop',
        'canonical_integer_answer': answer is not None,
        'single_confidence': len(confidence) == 1 and 0 <= int(confidence[0]) <= 100,
    }
    return {'accepted': all(checks.values()), 'checks': checks,
            'confidence': int(confidence[0]) if checks['single_confidence'] else -1}
