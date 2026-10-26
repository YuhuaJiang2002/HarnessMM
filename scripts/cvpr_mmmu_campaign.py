#!/usr/bin/env python3
"""Dependency-light protocol primitives for the full public MMMU campaign.

All benchmark access and native processor/model work are remote-only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import socket
from pathlib import Path

ARMS = ("base", "grounding", "generic", "format")
IMAGE_TAG = re.compile(r"<image\s+(\d+)>")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def load(path):
    return json.loads(Path(path).read_text())


def require_remote(config):
    if socket.gethostname() != config["hostname"]:
        raise RuntimeError("MMMU data, preparation, scoring and model calls require h200-1")


def options_list(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, list) or any(not isinstance(x, str) for x in options):
        raise ValueError("Expected a public list of option strings")
    return options


def reference_prompt(row, template):
    options = options_list(row["options"])
    if row["question_type"] == "multiple-choice":
        choices = "".join(f"({chr(65 + i)}) {answer}\n"
                          for i, answer in enumerate(options))
        prompt = template["multi_choice_example_format"].format(row["question"], choices)
    else:
        prompt = template["short_ans_example_format"].format(row["question"])
    task = template["task_instructions"]
    return task.strip() + "\n\n" + prompt if task else prompt


def image_content(prompt, images):
    """Keep text/tags byte-exact and bind numbered references to actual images."""
    content, consumed, position = [], set(), 0
    for match in IMAGE_TAG.finditer(prompt):
        index = int(match.group(1))
        if index not in images:
            raise ValueError(f"Missing public image_{index} referenced by question/options")
        content.append({"type": "text", "text": prompt[position:match.end()]})
        if index not in consumed:
            content.append({"type": "image", "image": images[index]})
            consumed.add(index)
        position = match.end()
    if position < len(prompt):
        content.append({"type": "text", "text": prompt[position:]})
    for index in sorted(images):
        if index not in consumed:
            content.extend([{"type": "text", "text": f"\n<image {index}>"},
                            {"type": "image", "image": images[index]}])
    return content


def messages(row, template, instruction, images):
    prompt = reference_prompt(row, template)
    if instruction:
        prompt = instruction + "\n\n" + prompt
    return [{"role": "user", "content": image_content(prompt, images)}]


def schedule(rows, config, split, arms=ARMS, phase="official"):
    seed = config["execution"]["schedule_seed"]
    ordered = sorted(rows, key=lambda row: (row["subject"], digest([seed, row["id"]])))
    positions, result = {}, []
    for row in ordered:
        subject = row["subject"]
        index = positions.get(subject, 0)
        positions[subject] = index + 1
        slot = config["resources"]["slots"][index % 2]["index"]
        rotation = int(digest([seed, row["id"], "arms"])[:8], 16) % len(arms)
        for arm in arms[rotation:] + arms[:rotation]:
            result.append({"id": row["id"], "subject": subject, "split": split,
                           "arm": arm, "slot": slot, "phase": phase,
                           "key": f"{phase}/{split}/{row['id']}/{arm}"})
    return result


def validate_population(rows, config, split, subjects):
    expected = config["benchmark"]["splits"][split]
    ids = [row["id"] for row in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Incomplete/duplicate {split}: {len(ids)} != {expected}")
    if len(subjects) != config["benchmark"]["subjects"]:
        raise ValueError("The official 30-subject population is required")
    per_subject = config["benchmark"]["counts_per_subject"][split]
    for subject in subjects:
        found = {row["id"] for row in rows if row["subject"] == subject}
        exact = {f"{split}_{subject}_{i}" for i in range(1, per_subject + 1)}
        if found != exact:
            raise ValueError(f"Unexpected full public split IDs for {split}/{subject}")


def reserve_plan(cells, rows_by_id, config):
    reservations = []
    for cell in cells:
        count = rows_by_id[cell["id"]]["prompts"][cell["arm"]]["prompt_tokens"]
        total = count + config["execution"]["max_tokens"]
        if total > config["model"]["max_model_len"]:
            raise ValueError(f"Native input exceeds context, no truncation allowed: {cell['key']}")
        reservations.append(total)
    if len(cells) > config["resources"]["maximum_model_requests"]:
        raise ValueError("Request reservation exceeds authorization")
    if sum(reservations) > config["resources"]["maximum_aggregate_tokens"]:
        raise ValueError("Token reservation exceeds authorization")
    return {"model_requests": len(cells), "all_in_tokens": sum(reservations),
            "output_tokens_per_request": config["execution"]["max_tokens"],
            "per_cell": dict(zip((cell["key"] for cell in cells), reservations))}
