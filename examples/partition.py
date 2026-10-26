"""Build a reproducible 20/20/60 partition of synthetic independent tasks."""
import json

from phiagent.harness.benchmark_partition import build_partition, canonical_sha256


def main():
    records = [
        {"id": f"example-{i:02d}",
         "payload_sha256": canonical_sha256({"synthetic_payload": i}),
         "public_task_sha256": canonical_sha256({"synthetic_public_task": i})}
        for i in range(10)
    ]
    partition = build_partition(records, benchmark="synthetic-example", seed=20261026,
                                independent_unit="task")
    print(json.dumps({"benchmark": partition["benchmark"],
                      "counts": partition["counts"],
                      "splits": partition["splits"]}, indent=2))


if __name__ == "__main__":
    main()
