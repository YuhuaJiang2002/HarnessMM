"""The prospective native app bank has the same policy for matched fast/slow arms."""


def validate_native_app_policy(policy, *, treatment, encoding):
    expected = {"version": policy.get("version") if isinstance(policy, dict) else None,
                "enabled": policy.get("enabled") if isinstance(policy, dict) else None,
                "maximum_candidates": 16, "maximum_fast_steps": 16,
                "maximum_public_bytes": 32768}
    if (not isinstance(policy, dict) or policy != expected
            or type(policy.get("version")) is not int or policy["version"] not in (3, 4)
            or type(policy.get("enabled")) is not bool or encoding != "native_text"
            or not isinstance(treatment, dict)
            or treatment.get("artifact_kind") != "training-derived"):
        raise ValueError("Native app policy needs the frozen TRAIN-derived treatment and original text")
    return dict(policy)
