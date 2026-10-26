"""Bounded, optional TACO-RL author-code adaptation; never an untrained baseline.

This module has no model, dataset, endpoint, or evaluator import-time operations.
The parent owns checkpoint loading, split validation, allocation and the training
loop. Long instructions use a deterministic, source-bound plan of whole-word
chunks, each at most 510 author tokens. Exact source whitespace between chunks is
protected; the author sampler still performs learned keep/drop within each chunk.
Every chunk is passed separately to the author's private __compress method with
an explicit XLM-R TokenClfDataset binding. Its categorical sampling statements are
unchanged; its high-level lossy/off-by-one chunker and cross-batch indexing are avoided.
One full-prompt reward is broadcast to all chunk graphs for one author update.

The update class is extracted, unchanged, from hash-verified author source rather
than importing its executable summarization training script. QA replaces the
author's summarization reward; it does not replace categorical sampling or
reward-weighted REINFORCE with a heuristic selector.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import string
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


IDENTITY = "TACO-RL author-code adaptation, instruction-only, frozen-Qwen reward actor"
AUTHOR_REVISION = "76b4b13704012b45dbf6fc3adb340055d9faa570"
INITIALIZER_REPOSITORY = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
INITIALIZER_REVISION = "ebaba9b0e874dadd3003ffcff828e4397e568089"
SCORER_REVISION = "b86bbcc63352786f0ca246b76191f062dce4b52b"
SCORER_SHA256 = "02a7855398e7133952ee374582fcc90a8b6dcf6216e0759347f29487074ee992"
IMPORT_ONLY_SAMPLER_SHA256 = "88f606375b5d2421a31bc018012ef2e33f705fdc5b39e47a210555236153bee5"
PATCHED_SAMPLER_SHA256 = "70f945be2a38f65613225fadfdee61248ad706acdc3554f584221de560216441"
AUTHOR_UPDATE_SHA256 = "54d54ddb2d368002202c2543b8a615a743d1b48a572b3e89669a251ef7b24380"
SOURCE_PATCH_ID = "taco-author-xlmr-compatibility-v2"
DATASET_BINDING_PATCH_ID = "taco-xlmr-input-dataset-binding-v1"
LONG_ADAPTER_ID = "taco-source-bound-long-instruction-v2"
INITIALIZER_LOAD_ID = "taco-repository-id-private-cache-v1"
MUTABLE_SCOPE_ID = "skill-section-including-inner-module-headings-v1"
RELEASE_SCHEMA = "taco_instruction_training_release_v5"
GRANT_SCHEMA = "taco_instruction_training_grant_v5"
TRAINING_BUDGET_REVISION = "taco-training-budget-2048-v1"
TRAINING_BUDGET_CAPS = {
    "max_updates": 2048,
    "max_classifier_calls": 65536,
    "max_reward_requests": 2048,
    "max_wall_seconds": 14400,
    "max_tokens_per_request": 65536,
    "max_total_reward_tokens": 16777216,
}
TRAINING_BUDGET_CAPS_SHA256 = "c6fd1b43ec6b82dd9ec492f6f264f2f7ff54ebd4c74d71961e8b1f1a932c116b"
CHUNKING = {
    "algorithm": "source-whitespace-greedy-v1",
    "max_author_tokens_per_chunk": 510,
    "max_chunks": 64,
    "max_source_characters": 131072,
    "separator_policy": "preserve_original_source_whitespace",
    "reward_policy": "one_full_prompt_reward_broadcast_to_all_chunks",
    "loss_reduction": "unchanged_author_mean_over_chunks_by_512_positions",
}
RELEASE_FIELDS = (
    "initializer", "instruction_profile", "training_split_manifest_sha256",
    "dependency_lock_sha256", "answer_parser_sha256", "qwen_provenance_sha256",
    "qwen_revision", "qwen_endpoint", "qwen_served_model", "decoding",
    "seed", "planned_epochs", "examples_per_epoch",
)
GRANT_FIELDS = (
    "schema", "authorized", "operation", "hostname", "run_id", "account",
    "physical_gpu_uuid", "parent_calibration_gate_passed", "parent_calibration_receipt_sha256",
    "budget", "release_sha256",
)
AUTHOR_HASHES = {
    "llmlingua/__init__.py":
        "0c900af1589cada85220123ca0c678f9ba5f0229c37c158f5eaf56b70bbeb616",
    "llmlingua/taco-rl/__init__.py":
        "b771b0d8c6bbf1bcf19b24d2b74b7171685a90d68d3a37763117715b20a2007a",
    "llmlingua/taco-rl/prompt_compressor_reinforce.py":
        "c49a551838a5e619ddf881e033b53346344fbf1f6a1eeef9c0fd04aa6dcfe2b6",
    "llmlingua/prompt_compressor.py":
        "08c2bd510fd46fe1427553e7f51436825d802068fac85b62a57efdcc423f811e",
    "llmlingua/utils.py":
        "4a4f96a063758a964410591ececb8f8f7b342452f16a0120281863500b058219",
    "experiments/taco-rl/utils.py":
        "3c7ad35d1c0349069f5ff1e9db479c1018a1ffe1c613f422171c410beb0d5373",
    "experiments/taco-rl/train_reinforce.py":
        "72223ae1095b77549030d98d75f3a7caf09f8b53c1814625816537aced6740d9",
    "experiments/taco-rl/configs/train_reinforce.yaml":
        "95ef358ce534dc030937589e280c9ec4ffdc61d0c7b2c70f8aede8185e625ec9",
    "setup.py": "dc1db5a6940ac538482036e7e9c016eacaf0fd31855c806c850b2a1f2bc74872",
    "LICENSE": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
}
HELPER_NAMES = (
    "TokenClfDataset", "get_pure_token", "is_begin_of_new_word", "replace_added_token",
)
OLD_IMPORT = (
    "from utils import TokenClfDataset, get_pure_token, "
    "is_begin_of_new_word, replace_added_token"
)
NEW_IMPORT = OLD_IMPORT.replace("from utils", "from llmlingua.utils")
UNBOUND_DATASET_ARGUMENTS = "chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len"
BOUND_DATASET_ARGUMENTS = UNBOUND_DATASET_ARGUMENTS + ", model_name=self.model_name"
XLMR_SPECIAL_TOKENS = {
    "bos_token": "<s>", "cls_token": "<s>",
    "eos_token": "</s>", "sep_token": "</s>",
    "pad_token": "<pad>", "unk_token": "<unk>", "mask_token": "<mask>",
}
SKILL_MARKER = "## Skill\n# Question Answering Skill\n\n"
CONTRACT_MARKER = "## Answer Contract\n"
ACCOUNT = "taco_baseline_acquisition"


class ReadinessError(RuntimeError):
    """A missing prerequisite, unsupported author branch, or failed operation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    return digest(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_pins(config: Mapping[str, Any]) -> None:
    require(
        config.get("identity") == IDENTITY
        and config.get("author", {}).get("revision") == AUTHOR_REVISION
        and config.get("author", {}).get("source_patch_id") == SOURCE_PATCH_ID
        and config.get("author", {}).get("dataset_binding_patch_id") == DATASET_BINDING_PATCH_ID
        and config.get("author", {}).get("patched_sampler_sha256") == PATCHED_SAMPLER_SHA256
        and config.get("author", {}).get("unchanged_extracted_update_sha256") == AUTHOR_UPDATE_SHA256
        and config.get("initializer", {}).get("repository") == INITIALIZER_REPOSITORY
        and config.get("initializer", {}).get("revision") == INITIALIZER_REVISION
        and config.get("official_scoring", {}).get("source_sha256") == SCORER_SHA256
        and config.get("official_scoring", {}).get("source_revision") == SCORER_REVISION,
        "frozen baseline identity or source/checkpoint/scorer provenance changed",
    )
    require(
        config.get("training_budget_caps") == TRAINING_BUDGET_CAPS
        and json_digest(config["training_budget_caps"]) == TRAINING_BUDGET_CAPS_SHA256
        and config.get("training_budget_revision", {}).get("id") == TRAINING_BUDGET_REVISION
        and config["training_budget_revision"].get("caps_sha256") == TRAINING_BUDGET_CAPS_SHA256,
        "prospective training-budget revision or its fixed caps changed",
    )
    settings = config.get("adaptation", {})
    require(
        all(settings.get(key) == value for key, value in {
            "batch_size": 1, "policy_lr": 1e-5, "entropy_coeff": 0.01,
            "compression_rate": 0.5, "compression_relaxation_tokens": 30,
            "author_max_sequence_length": 512, "max_author_tokens_per_chunk": 510,
            "max_chunks_per_instruction": CHUNKING["max_chunks"],
            "max_instruction_characters": CHUNKING["max_source_characters"],
            "mutable_scope_id": MUTABLE_SCOPE_ID,
        }.items()),
        "author defaults, mutable scope or long-instruction limits changed; a new protocol is required",
    )


def checked_source(path: Path, expected_hash: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular source: {path}")
    require(path.stat().st_size <= 2 * 1024**2, "source exceeds the bounded source size")
    raw = path.read_bytes()
    require(digest(raw) == expected_hash, f"source SHA256 mismatch: {path}")
    return raw.decode("utf-8")


def definition_source(source: str, names: Sequence[str]) -> str:
    """Select exact top-level definitions, not imports or executable top-level code."""
    nodes = ast.parse(source).body
    selected = []
    for name in names:
        matches = [
            node for node in nodes
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
        ]
        require(len(matches) == 1, f"expected exactly one definition of {name}")
        require(not matches[0].decorator_list, f"unexpected decorator on {name}")
        selected.append(ast.get_source_segment(source, matches[0]))
    return "\n\n".join(selected) + "\n"


def loopback_chat_url(endpoint: str) -> str:
    require(isinstance(endpoint, str), "reward endpoint must be a string")
    try:
        parts = urlsplit(endpoint)
        port = parts.port
    except ValueError as error:
        raise ReadinessError("invalid loopback reward endpoint") from error
    require(
        parts.scheme == "http" and parts.hostname in {"127.0.0.1", "::1"}
        and port is not None and port > 0 and parts.username is None and parts.password is None
        and parts.path.rstrip("/") == "/v1" and not parts.query and not parts.fragment,
        "reward endpoint must be an explicit HTTP loopback address/port ending in /v1",
    )
    return endpoint.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class InstructionEnvelope:
    """Inner module headings/rules are editable; the outer skill heading is protected.

    The span starts after SKILL_MARKER and ends before the two-newline separator
    preceding CONTRACT_MARKER. The complete user question/context is immutable.
    """

    system: str
    user: str
    source_sha256: str
    instruction_start: int
    instruction_end: int

    def __post_init__(self) -> None:
        require(
            isinstance(self.system, str) and isinstance(self.user, str) and bool(self.user)
            and is_sha256(self.source_sha256)
            and digest(self.system.encode()) == self.source_sha256,
            "immutable envelope requires the exact frozen source hash and complete user text",
        )
        require(
            self.system.count(SKILL_MARKER) == self.system.count(CONTRACT_MARKER) == 1
            and self.instruction_start == self.system.index(SKILL_MARKER) + len(SKILL_MARKER)
            and self.instruction_end == self.system.index(CONTRACT_MARKER) - 2
            and self.system[self.instruction_end:self.instruction_end + 2] == "\n\n"
            and self.instruction_start < self.instruction_end,
            "instruction span must exclude the prefix, contract and its two-newline separator",
        )

    @classmethod
    def from_messages(
        cls, messages: Sequence[Mapping[str, str]], *, expected_system_sha256: str,
    ) -> InstructionEnvelope:
        require(len(messages) == 2, "the verified surface supports system + user only")
        require(
            all(set(message) == {"role", "content"} for message in messages)
            and [message["role"] for message in messages] == ["system", "user"]
            and all(isinstance(message["content"], str) for message in messages),
            "unsupported message envelope; no fields may be silently discarded",
        )
        system, user = (message["content"] for message in messages)
        require(
            is_sha256(expected_system_sha256)
            and digest(system.encode("utf-8")) == expected_system_sha256,
            "frozen instruction source SHA256 mismatch",
        )
        require(
            system.count(SKILL_MARKER) == system.count(CONTRACT_MARKER) == 1,
            "unique skill and answer-contract boundaries are required",
        )
        start = system.index(SKILL_MARKER) + len(SKILL_MARKER)
        end = system.index(CONTRACT_MARKER) - 2
        require(start < end and bool(user), "empty/reversed instruction or user message")
        return cls(system, user, expected_system_sha256, start, end)

    @property
    def instruction(self) -> str:
        return self.system[self.instruction_start:self.instruction_end]

    def mutable_span_receipt(self) -> dict[str, Any]:
        return {
            "schema": "taco_instruction_mutable_span_v1", "scope_id": MUTABLE_SCOPE_ID,
            "source_system_sha256": self.source_sha256,
            "span_frame": "original_system_prompt_unicode_codepoints",
            "bounds": "zero_based_half_open",
            "start": self.instruction_start, "end": self.instruction_end,
            "mutable_text_sha256": digest(self.instruction.encode()),
            "protected_prefix_sha256": digest(self.system[:self.instruction_start].encode()),
            "protected_suffix_sha256": digest(self.system[self.instruction_end:].encode()),
            "inner_module_headings_editable": True,
            "inner_rule_bodies_editable": True,
            "outer_question_answering_skill_heading_editable": False,
            "skill_wrapper_heading_editable": False,
            "answer_contract_or_its_separator_editable": False,
            "user_question_or_context_editable": False,
        }

    def render(self, compressed_instruction: str) -> list[dict[str, str]]:
        require(isinstance(compressed_instruction, str), "compression must return text")
        return [
            {
                "role": "system",
                "content": self.system[:self.instruction_start] + compressed_instruction
                + self.system[self.instruction_end:],
            },
            {"role": "user", "content": self.user},
        ]


@dataclass(frozen=True)
class InstructionChunk:
    start: int
    end: int
    author_tokens: int


@dataclass(frozen=True)
class InstructionChunkPlan:
    source: str
    system_sha256: str
    tokenizer_manifest_sha256: str
    chunks: tuple[InstructionChunk, ...]
    tokenizer_calls: int
    max_author_tokens: int

    def __post_init__(self) -> None:
        require(
            is_sha256(self.system_sha256) and is_sha256(self.tokenizer_manifest_sha256)
            and 0 < len(self.chunks) <= CHUNKING["max_chunks"]
            and 0 < len(self.source) <= CHUNKING["max_source_characters"]
            and type(self.max_author_tokens) is int and 0 < self.max_author_tokens <= 510,
            "invalid source-bound chunk plan",
        )
        previous = 0
        for chunk in self.chunks:
            require(
                type(chunk.author_tokens) is int and 0 < chunk.author_tokens <= self.max_author_tokens
                and previous <= chunk.start < chunk.end <= len(self.source)
                and not self.source[chunk.start].isspace()
                and not self.source[chunk.end - 1].isspace()
                and not self.source[previous:chunk.start].strip()
                and (chunk.start == 0 or self.source[chunk.start - 1].isspace())
                and (chunk.end == len(self.source) or self.source[chunk.end].isspace()),
                "chunk plan omits, overlaps, splits a word, or exceeds the author token limit",
            )
            previous = chunk.end
        require(not self.source[previous:].strip(), "chunk plan omits a source suffix")

    def texts(self) -> tuple[str, ...]:
        return tuple(self.source[chunk.start:chunk.end] for chunk in self.chunks)

    def recompose(self, compressed_chunks: Sequence[str]) -> str:
        require(
            not isinstance(compressed_chunks, str) and len(compressed_chunks) == len(self.chunks)
            and all(isinstance(text, str) for text in compressed_chunks),
            "exactly one text result per source chunk is required",
        )
        output = []
        previous = 0
        for chunk, compressed in zip(self.chunks, compressed_chunks):
            output.extend((self.source[previous:chunk.start], compressed))
            previous = chunk.end
        output.append(self.source[previous:])
        return "".join(output)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "taco_instruction_chunk_plan_v2", "adapter_id": LONG_ADAPTER_ID,
            "mutable_scope_id": MUTABLE_SCOPE_ID,
            "algorithm": CHUNKING["algorithm"], "source_system_sha256": self.system_sha256,
            "source_instruction_sha256": digest(self.source.encode()),
            "source_instruction_characters": len(self.source),
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "max_author_tokens_per_chunk": self.max_author_tokens,
            "span_frame": "instruction_unicode_codepoints", "bounds": "zero_based_half_open",
            "planner_tokenizer_calls": self.tokenizer_calls,
            "chunks": [
                {
                    "index": index, "start": chunk.start, "end": chunk.end,
                    "utf8_start": len(self.source[:chunk.start].encode()),
                    "utf8_end": len(self.source[:chunk.end].encode()),
                    "author_tokens": chunk.author_tokens,
                    "text_sha256": digest(self.source[chunk.start:chunk.end].encode()),
                }
                for index, chunk in enumerate(self.chunks)
            ],
            "separators": [
                {"start": start, "end": end, "sha256": digest(self.source[start:end].encode())}
                for start, end in zip(
                    [0, *(chunk.end for chunk in self.chunks)],
                    [*(chunk.start for chunk in self.chunks), len(self.source)],
                )
            ],
            "input_recomposition_sha256": digest(self.recompose(self.texts()).encode()),
            "input_characters_dropped": 0, "input_normalization": "none",
            "classifier_calls_per_example": len(self.chunks),
            "reward_calls_per_example_upper_bound": 1,
            "policy_updates_per_example": 1,
        }


def plan_instruction_chunks(
    envelope: InstructionEnvelope, *, count_tokens: Callable[[str], int],
    tokenizer_manifest_sha256: str, max_author_tokens: int = 510,
) -> InstructionChunkPlan:
    """Pure planner; real tokenizer callbacks belong on the authorized remote host.

    No monotonic-token-count assumption, token decode/re-encode, truncation, or
    word splitting is used. One overlong whitespace-free unit fails explicitly.
    """
    source = envelope.instruction
    require(
        len(source) <= CHUNKING["max_source_characters"]
        and type(max_author_tokens) is int and 0 < max_author_tokens <= 510,
        "instruction/chunk size exceeds the declared bounded adapter",
    )
    words = list(re.finditer(r"\S+", source))
    require(bool(words), "instruction must contain at least one non-whitespace unit")
    calls = 0

    def count(text: str) -> int:
        nonlocal calls
        calls += 1
        value = count_tokens(text)
        require(type(value) is int and value > 0, "invalid author tokenizer count")
        return value

    chunks = []
    start, end, tokens = words[0].start(), words[0].end(), count(words[0].group())
    require(tokens <= max_author_tokens, "single source word exceeds the chunk limit; no truncation")
    for word in words[1:]:
        candidate_tokens = count(source[start:word.end()])
        if candidate_tokens <= max_author_tokens:
            end, tokens = word.end(), candidate_tokens
        else:
            chunks.append(InstructionChunk(start, end, tokens))
            require(len(chunks) < CHUNKING["max_chunks"], "instruction exceeds chunk-count bound")
            start, end, tokens = word.start(), word.end(), count(word.group())
            require(tokens <= max_author_tokens, "single source word exceeds the chunk limit; no truncation")
    chunks.append(InstructionChunk(start, end, tokens))
    return InstructionChunkPlan(
        source, envelope.source_sha256, tokenizer_manifest_sha256, tuple(chunks), calls,
        max_author_tokens,
    )


@dataclass(frozen=True)
class ChunkPolicySample:
    index: int
    text: str
    log_probs: Any
    entropy: Any


def author_chunk_sample(index: int, output: Any) -> ChunkPolicySample:
    """Unpack the verified author's eight-item private sampler return, without detaching."""
    require(isinstance(output, tuple) and len(output) == 8, "author sampler return arity changed")
    require(
        all(len(output[position]) == 1 for position in (0, 3, 4, 6, 7))
        and output[0][0] == output[6][0] and isinstance(output[6][0], str),
        "each call must produce one aligned author chunk; never zip/truncate partial outputs",
    )
    return ChunkPolicySample(index, output[6][0], output[4][0], output[7][0])


def validate_chunk_samples(
    plan: InstructionChunkPlan, samples: Sequence[ChunkPolicySample],
) -> None:
    require(
        len(samples) == len(plan.chunks)
        and [sample.index for sample in samples] == list(range(len(plan.chunks))),
        "missing, reordered or duplicate chunk policy graphs",
    )
    require(
        all(
            tensor.requires_grad is True and tuple(tensor.shape) == (512,)
            for sample in samples for tensor in (sample.log_probs, sample.entropy)
        ),
        "all author chunk graphs must remain differentiable, padded 512-position vectors",
    )
    require(
        len({id(sample.log_probs) for sample in samples}) == len(samples)
        and len({id(sample.entropy) for sample in samples}) == len(samples),
        "a chunk graph was reused instead of sampling every source chunk",
    )


def chunk_update_arguments(
    plan: InstructionChunkPlan, samples: Sequence[ChunkPolicySample], *,
    reward: float, compression_ratio: float,
) -> dict[str, Any]:
    validate_chunk_samples(plan, samples)
    require(
        type(reward) in (int, float) and math.isfinite(reward)
        and type(compression_ratio) in (int, float)
        and math.isfinite(compression_ratio) and compression_ratio > 0,
        "nonfinite full-prompt reward or compression ratio",
    )
    return {
        # The unchanged author mean already divides by C*512. Do not divide reward by C.
        "rewards_and_comp_ratios": [
            {"reward": float(reward), "comp_ratio": float(compression_ratio)} for _ in samples
        ],
        "old_log_probs": [sample.log_probs for sample in samples],
        "entropies": [sample.entropy for sample in samples],
    }


def validate_xlmr_special_tokens(tokenizer: Any) -> tuple[int, int, int]:
    require(
        all(getattr(tokenizer, name, None) == token for name, token in XLMR_SPECIAL_TOKENS.items()),
        "pinned XLM-R BOS/CLS, EOS/SEP, PAD, UNK and MASK spellings are required",
    )
    ids = tuple(tokenizer.convert_tokens_to_ids(["<s>", "</s>", "<pad>"]))
    require(
        len(ids) == 3 and all(type(value) is int and value >= 0 for value in ids)
        and len(set(ids)) == 3 and type(getattr(tokenizer, "unk_token_id", None)) is int
        and tokenizer.unk_token_id not in ids
        and ids == (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        and tokenizer.cls_token_id == tokenizer.bos_token_id
        and tokenizer.sep_token_id == tokenizer.eos_token_id,
        "XLM-R special-token IDs must resolve to distinct real vocabulary entries, not UNK",
    )
    return ids


def validate_xlmr_classifier_input(
    tokenizer: Any, input_ids: Any, attention_mask: Any, *, author_tokens: int,
) -> None:
    """Check the actual dataset output, before any classifier forward computation."""
    bos, eos, pad = validate_xlmr_special_tokens(tokenizer)
    require(
        type(author_tokens) is int and 0 < author_tokens <= 510
        and tuple(input_ids.shape) == tuple(attention_mask.shape) == (1, 512),
        "author dataset truncated or changed the complete planned chunk shape",
    )
    active = author_tokens + 2
    require(
        int(input_ids[0, 0].item()) == bos
        and int(input_ids[0, active - 1].item()) == eos
        and bool((input_ids[0, active:] == pad).all().item())
        and bool((attention_mask[0, :active] == 1).all().item())
        and bool((attention_mask[0, active:] == 0).all().item()),
        "author dataset must emit XLM-R BOS/EOS/PAD IDs and exact untruncated right-padding masks",
    )


def validate_initializer_metadata(initializer: Mapping[str, Any]) -> None:
    require(
        isinstance(initializer, dict) and set(initializer) == {
            "repository", "revision", "local_model_path", "receipt_sha256",
            "files_manifest_sha256", "tokenizer_manifest_sha256", "byte_verified", "transport",
            "hf_cache_dir", "private_hf_cache", "snapshot_alias_receipt_sha256",
        } and initializer["repository"] == INITIALIZER_REPOSITORY
        and initializer["revision"] == INITIALIZER_REVISION and initializer["byte_verified"] is True
        and initializer["transport"] in {"parent_byte_verified_public_mirror", "huggingface_snapshot"}
        and initializer["private_hf_cache"] is True
        and all(
            isinstance(initializer[key], str) and Path(initializer[key]).is_absolute()
            and ".." not in Path(initializer[key]).parts
            for key in ("local_model_path", "hf_cache_dir")
        ) and all(is_sha256(initializer[key]) for key in (
            "receipt_sha256", "files_manifest_sha256", "tokenizer_manifest_sha256",
            "snapshot_alias_receipt_sha256",
        )),
        "exact byte-verified initializer and a receipt-bound private HF cache are required",
    )
    physical, cache = Path(initializer["local_model_path"]), Path(initializer["hf_cache_dir"])
    require(
        physical != cache and physical not in cache.parents,
        "private cache must not be created inside the immutable initializer directory",
    )


def initializer_snapshot_path(initializer: Mapping[str, Any]) -> Path:
    """Derive the exact HF cache key without resolving/reading model files."""
    validate_initializer_metadata(initializer)
    return (
        Path(initializer["hf_cache_dir"])
        / ("models--" + INITIALIZER_REPOSITORY.replace("/", "--"))
        / "snapshots" / INITIALIZER_REVISION
    )


def pinned_compressor_kwargs(release: Mapping[str, Any]) -> dict[str, Any]:
    """Pure constructor arguments, NOT a model loader or cache/environment mutation.

    Keep model_name as the real repository ID: the author dispatches word
    boundaries on that string. A generic physical directory named 'model' is
    valid storage, but must never be passed as model_name.
    """
    require(
        release.get("schema") == RELEASE_SCHEMA
        and release.get("initializer_loading_id") == INITIALIZER_LOAD_ID,
        "repository-name loading requires the current frozen release schema",
    )
    initializer = release["initializer"]
    validate_initializer_metadata(initializer)
    return {
        "model_name": INITIALIZER_REPOSITORY,
        "model_config": {
            "revision": INITIALIZER_REVISION, "cache_dir": initializer["hf_cache_dir"],
            "local_files_only": True, "trust_remote_code": False,
        },
        "use_llmlingua2": True, "device_map": "cuda",
    }


def verify_runtime_initializer_binding(
    lease: RemoteTrainingLease, compressor: Any,
) -> dict[str, Any]:
    """Remote read-only identity/alias check; tensor-to-safetensors audit stays parent-owned."""
    lease.check()
    initializer = lease.authority["initializer"]
    snapshot = initializer_snapshot_path(initializer)
    physical = Path(initializer["local_model_path"])
    require(
        compressor.model_name == INITIALIZER_REPOSITORY
        and getattr(compressor.model.config, "_commit_hash", None) in (None, INITIALIZER_REVISION)
        and getattr(compressor.model.config, "num_labels", None) == 2
        and getattr(compressor.model.config, "model_type", None) == "xlm-roberta",
        "load with the exact repository ID, not a local-path model_name or substitute model",
    )
    require(
        snapshot.is_dir() and physical.is_dir()
        and snapshot.resolve(strict=True) == physical.resolve(strict=True),
        "private snapshots/<exact_revision> directory alias must resolve to the released model path",
    )
    return {
        "initializer_loading_id": INITIALIZER_LOAD_ID,
        "model_name": INITIALIZER_REPOSITORY, "revision": INITIALIZER_REVISION,
        "physical_model_path": str(physical), "hf_cache_dir": initializer["hf_cache_dir"],
        "snapshot_path": str(snapshot), "resolved_snapshot_path": str(snapshot.resolve(strict=True)),
        "snapshot_alias_receipt_sha256": initializer["snapshot_alias_receipt_sha256"],
        "files_manifest_sha256": initializer["files_manifest_sha256"],
        "local_files_only": True, "trust_remote_code": False,
        "tensor_verification_owner": "parent; compare original loaded tensors against safetensors",
        "tensor_verification_performed_by_this_check": False,
    }


def prepare_training_release(config: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """Static metadata preparation only; no files, models or datasets are acquired."""
    validate_pins(config)
    require(set(fields) == set(RELEASE_FIELDS), f"release fields must be exactly {RELEASE_FIELDS}")
    profile = fields["instruction_profile"]
    require(profile in config["instruction_source_sha256"], "unknown frozen instruction profile")
    release = {
        **json.loads(json.dumps(fields, allow_nan=False)),
        "schema": RELEASE_SCHEMA, "identity": IDENTITY,
        "author_revision": AUTHOR_REVISION, "source_patch_id": SOURCE_PATCH_ID,
        "dataset_binding_patch_id": DATASET_BINDING_PATCH_ID, "mutable_scope_id": MUTABLE_SCOPE_ID,
        "adapter_id": LONG_ADAPTER_ID, "adapter_source_sha256": digest(Path(__file__).read_bytes()),
        "initializer_loading_id": INITIALIZER_LOAD_ID,
        "patched_sampler_sha256": PATCHED_SAMPLER_SHA256,
        "author_update_sha256": AUTHOR_UPDATE_SHA256,
        "instruction_source_sha256": config["instruction_source_sha256"][profile],
        "scorer_sha256": SCORER_SHA256, "qwen_repository": "Qwen/Qwen3.8-27B",
        "qwen_frozen": True, "chunking": dict(CHUNKING),
        "adaptation_sha256": json_digest(config["adaptation"]),
        "training_budget_revision": TRAINING_BUDGET_REVISION,
        "training_budget_caps_sha256": TRAINING_BUDGET_CAPS_SHA256,
    }
    require(
        type(fields["planned_epochs"]) is int and fields["planned_epochs"] > 0
        and type(fields["examples_per_epoch"]) is int and fields["examples_per_epoch"] > 0,
        "explicit positive epoch and training-example counts are required",
    )
    release["planned_total_updates"] = fields["planned_epochs"] * fields["examples_per_epoch"]
    validate_training_release(config, release)
    return release


def validate_training_release(config: Mapping[str, Any], release: Mapping[str, Any]) -> None:
    validate_pins(config)
    fixed = {
        "schema": RELEASE_SCHEMA, "identity": IDENTITY,
        "author_revision": AUTHOR_REVISION, "source_patch_id": SOURCE_PATCH_ID,
        "dataset_binding_patch_id": DATASET_BINDING_PATCH_ID, "mutable_scope_id": MUTABLE_SCOPE_ID,
        "adapter_id": LONG_ADAPTER_ID, "adapter_source_sha256": digest(Path(__file__).read_bytes()),
        "initializer_loading_id": INITIALIZER_LOAD_ID,
        "patched_sampler_sha256": PATCHED_SAMPLER_SHA256,
        "author_update_sha256": AUTHOR_UPDATE_SHA256, "scorer_sha256": SCORER_SHA256,
        "qwen_repository": "Qwen/Qwen3.8-27B", "qwen_frozen": True,
        "chunking": CHUNKING, "adaptation_sha256": json_digest(config["adaptation"]),
        "training_budget_revision": TRAINING_BUDGET_REVISION,
        "training_budget_caps_sha256": TRAINING_BUDGET_CAPS_SHA256,
    }
    require(
        set(release) == set(RELEASE_FIELDS) | set(fixed) | {
            "instruction_source_sha256", "planned_total_updates",
        } and all(release.get(key) == value for key, value in fixed.items()),
        "training release schema, source identity, chunking, adaptation or budget binding changed",
    )
    validate_initializer_metadata(release["initializer"])
    require(
        release["instruction_profile"] in config["instruction_source_sha256"]
        and release["instruction_source_sha256"] == config["instruction_source_sha256"][
            release["instruction_profile"]
        ] and all(is_sha256(release[key]) for key in (
            "training_split_manifest_sha256", "dependency_lock_sha256",
            "answer_parser_sha256", "qwen_provenance_sha256",
        )),
        "release profile or frozen input/actor/parser/environment receipts are missing",
    )
    require(
        type(release["seed"]) is int and 0 <= release["seed"] < 2**32
        and all(type(release[key]) is int and release[key] > 0 for key in (
            "planned_epochs", "examples_per_epoch", "planned_total_updates",
        )) and release["planned_total_updates"] == (
            release["planned_epochs"] * release["examples_per_epoch"]
        ) and release["planned_total_updates"] <= config["training_budget_caps"]["max_updates"],
        "release seed or complete training schedule is invalid",
    )
    require(
        isinstance(release["qwen_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", release["qwen_revision"]) is not None
        and isinstance(release["qwen_served_model"], str) and bool(release["qwen_served_model"]),
        "freeze the exact Qwen revision and served model ID",
    )
    loopback_chat_url(release["qwen_endpoint"])
    decoding = release["decoding"]
    require(
        isinstance(decoding, dict) and type(decoding.get("max_tokens")) is int
        and 0 < decoding["max_tokens"] < config["training_budget_caps"]["max_tokens_per_request"]
        and not (set(decoding) & {"model", "messages", "stream", "tools"}),
        "release decoding must freeze an output cap without replacing protected request fields",
    )


def resolve_training_grant(
    config: Mapping[str, Any], release: Mapping[str, Any], grant: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind pure release metadata to a separate explicit execution grant."""
    validate_training_release(config, release)
    require(
        set(grant) == set(GRANT_FIELDS) and grant["schema"] == GRANT_SCHEMA
        and grant["release_sha256"] == json_digest(release)
        and grant["authorized"] is True and grant["operation"] == "train-author-adaptation"
        and grant["account"] == ACCOUNT and grant["parent_calibration_gate_passed"] is True
        and is_sha256(grant["parent_calibration_receipt_sha256"])
        and isinstance(grant["hostname"], str) and bool(grant["hostname"])
        and isinstance(grant["run_id"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", grant["run_id"]) is not None
        and isinstance(grant["physical_gpu_uuid"], str)
        and re.fullmatch(r"GPU-[0-9a-fA-F-]{16,}", grant["physical_gpu_uuid"]) is not None,
        "exact training grant schema and canonical release SHA256 are required",
    )
    budget = grant["budget"]
    require(
        isinstance(budget, dict) and set(budget) == set(config["training_budget_caps"])
        and all(
            type(budget[key]) is int and 0 < budget[key] <= cap
            for key, cap in config["training_budget_caps"].items()
        ) and budget["max_updates"] == release["planned_total_updates"]
        and budget["max_reward_requests"] >= release["planned_total_updates"]
        and budget["max_total_reward_tokens"] >= budget["max_tokens_per_request"]
        and release["decoding"]["max_tokens"] < budget["max_tokens_per_request"],
        "grant must fund a complete frozen schedule with finite separate classifier/reward budgets",
    )
    return json.loads(json.dumps({**release, **grant}, allow_nan=False))


def compression_reward(
    original_tokens: int, compressed_tokens: int, official_f1: float | None,
    *, rate: float, relaxation_tokens: int,
) -> float:
    """Author reward gates, with official QA F1 replacing summarization BLEU.

    Zero-length outputs hit an author division-by-zero branch, so they block
    rather than receiving an invented ratio. Missing/failed responses never
    receive the author's positive +0.4 missing-output fallback.
    """
    require(
        type(original_tokens) is int and type(compressed_tokens) is int
        and original_tokens > 0 and compressed_tokens > 0,
        "zero/invalid token counts: author compression-ratio branch is unsupported",
    )
    require(
        type(rate) in (int, float) and math.isfinite(rate) and 0 < rate < 1
        and type(relaxation_tokens) is int and relaxation_tokens >= 0,
        "unsupported compression reward settings",
    )
    if compressed_tokens <= 0.1 * original_tokens:
        return -0.4
    require(
        type(official_f1) in (int, float) and math.isfinite(official_f1)
        and 0 <= official_f1 <= 1,
        "a successful, unscaled official MRQA F1 in [0, 1] is required",
    )
    if abs(compressed_tokens - rate * original_tokens) > relaxation_tokens:
        return -0.1
    return float(official_f1)


def validate_remote_authority(
    authority: Mapping[str, Any], run_directory: Path, *, operation: str,
) -> Path:
    """Pure checks precede directory creation, downloads, GPU inspection or imports."""
    require(platform.system() == "Linux", "remote Linux host required; local runs are forbidden")
    require(
        authority.get("authorized") is True and authority.get("operation") == operation
        and authority.get("hostname") == socket.gethostname(),
        "explicit parent authorization for this operation and exact host is required",
    )
    require(authority.get("account") == ACCOUNT, "baseline acquisition must be separately charged")
    path = run_directory.absolute()
    root = Path.cwd().absolute() / "runs"
    require(
        path.parent == root and root.resolve() == root and path.resolve() == path
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", path.name) is not None
        and authority.get("run_id") == path.name and not path.exists(),
        "a fresh, nonsymlink runs/<authorized-run-id> directory is required",
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], capture_output=True, timeout=10, check=False,
    )
    require(ignored.returncode == 0, "run directory must be ignored by git")
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class RemoteTrainingLease:
    """Explicit parent-owned remote budget; construction is not a launch command."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ReadinessError("use RemoteTrainingLease.open with a complete parent authorization")

    @classmethod
    def open(
        cls, config: Mapping[str, Any], authority: Mapping[str, Any], run_directory: Path,
        *, release: Mapping[str, Any] | None = None,
    ) -> RemoteTrainingLease:
        require(config.get("training_launch_enabled") is True, "training is disabled in readiness config")
        validate_pins(config)
        path = validate_remote_authority(authority, run_directory, operation="train-author-adaptation")
        require(isinstance(release, Mapping), "an immutable training release is required")
        grant = dict(authority)
        authority = resolve_training_grant(config, release, grant)
        require("torch" not in sys.modules, "GPU selection must precede importing torch")
        require(
            authority.get("parent_calibration_gate_passed") is True
            and all(is_sha256(authority.get(key)) for key in (
                "parent_calibration_receipt_sha256", "training_split_manifest_sha256",
                "dependency_lock_sha256", "qwen_provenance_sha256", "answer_parser_sha256",
            )),
            "parent's unchanged passing calibration, split/lock/parser/Qwen receipts are required",
        )
        budget = authority.get("budget", {})
        for key, maximum in config["training_budget_caps"].items():
            require(
                type(budget.get(key)) is int and 0 < budget[key] <= maximum,
                f"explicit positive bounded {key} is required",
            )
        require(type(authority.get("seed")) is int, "an explicit training seed is required")
        require(0 <= authority["seed"] < 2**32, "seed must fit numpy's unsigned 32-bit range")
        require(
            authority.get("instruction_profile") in config["instruction_source_sha256"],
            "freeze exactly one instruction profile before training",
        )
        require(
            type(authority.get("planned_epochs")) is int and authority["planned_epochs"] > 0
            and budget["max_updates"] % authority["planned_epochs"] == 0,
            "freeze complete epochs and the corresponding total planned update count",
        )
        require(
            re.fullmatch(r"GPU-[0-9a-fA-F-]{16,}", authority.get("physical_gpu_uuid", "")) is not None,
            "parent must allocate a physical GPU UUID",
        )
        require(
            authority.get("qwen_repository") == "Qwen/Qwen3.8-27B"
            and re.fullmatch(r"[0-9a-f]{40}", authority.get("qwen_revision", "")) is not None
            and authority.get("qwen_frozen") is True,
            "exact frozen Qwen3.8-27B repository/revision provenance is required",
        )
        loopback_chat_url(authority.get("qwen_endpoint", ""))
        require(bool(authority.get("qwen_served_model")), "exact served model ID is required")
        require(
            isinstance(authority.get("decoding"), dict)
            and type(authority["decoding"].get("max_tokens")) is int
            and 0 < authority["decoding"]["max_tokens"] < budget["max_tokens_per_request"]
            and not (set(authority["decoding"]) & {"model", "messages", "stream", "tools"}),
            "freeze decoding/output cap without overriding protected request fields",
        )
        path.parent.mkdir(exist_ok=True)
        path.mkdir()
        lease = object.__new__(cls)
        lease.root = path
        lease.config = json.loads(json.dumps(config, allow_nan=False))
        lease.authority = json.loads(json.dumps(authority, allow_nan=False))
        lease.frozen_binding = lease.binding()
        lease.started = time.monotonic()
        lease.requests = lease.updates = lease.tokens = 0
        lease.classifier_calls = lease.completed_classifier_calls = 0
        lease.unsettled_requests: set[int] = set()
        lease.failed = False
        try:
            write_json(path / "config.json", config)
            write_json(path / "authorization.json", grant)
            write_json(path / "release.json", release)
            write_json(path / "resolved_authority.json", authority)
            (path / "taco_instruction_adapter.py").write_bytes(Path(__file__).read_bytes())
            write_json(path / "environment.json", {
                "hostname": socket.gethostname(), "python": sys.version, "command": sys.argv,
                "seed": authority["seed"], "account": ACCOUNT,
                "packages": sorted(
                    (dist.metadata["Name"], dist.version)
                    for dist in importlib.metadata.distributions()
                ),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                ),
                "adapter_sha256": digest(Path(__file__).read_bytes()),
            })
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                 "--format=csv,noheader,nounits"], text=True, timeout=15,
            )
            rows = [line.split(",") for line in gpu.splitlines()]
            selected = [
                [part.strip() for part in row] for row in rows
                if len(row) == 5 and row[1].strip() == authority["physical_gpu_uuid"]
            ]
            require(len(selected) == 1, "allocated physical GPU is absent or ambiguous")
            os.environ["CUDA_VISIBLE_DEVICES"] = authority["physical_gpu_uuid"]
            write_json(path / "gpu_selection.json", {
                "query": gpu, "selected": selected[0],
                "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                "allocation_owner": "parent", "inspected_at_unix": time.time(),
            })
        except Exception as error:
            lease.fail(error)
            raise
        return lease

    def binding(self) -> str:
        return digest(json.dumps(
            [self.config, self.authority], sort_keys=True, allow_nan=False,
        ).encode())

    def check(self) -> None:
        require(not self.failed, "failed lease cannot be resumed or silently retried")
        require(self.binding() == self.frozen_binding, "frozen configuration/authority changed")
        require(
            platform.system() == "Linux" and socket.gethostname() == self.authority["hostname"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == self.authority["physical_gpu_uuid"],
            "remote host or physical GPU binding changed",
        )
        require(
            time.monotonic() - self.started < self.authority["budget"]["max_wall_seconds"],
            "baseline wall-time budget exhausted",
        )

    def event(self, value: Mapping[str, Any]) -> None:
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "account": ACCOUNT, "elapsed_seconds": time.monotonic() - self.started,
                "requests": self.requests, "updates": self.updates, "tokens": self.tokens,
                "classifier_calls": self.classifier_calls,
                "completed_classifier_calls": self.completed_classifier_calls,
                **value,
            }, sort_keys=True, allow_nan=False) + "\n")

    def fail(self, error: Exception) -> None:
        self.failed = True
        self.event({"status": "failed", "error_type": type(error).__name__, "error": str(error)})

    def reserve_request(self) -> int:
        self.check()
        budget = self.authority["budget"]
        require(self.requests < budget["max_reward_requests"], "reward request budget exhausted")
        require(
            self.tokens + budget["max_tokens_per_request"] <= budget["max_total_reward_tokens"],
            "insufficient reward token budget for the full per-request reservation",
        )
        self.requests += 1
        self.tokens += budget["max_tokens_per_request"]
        self.unsettled_requests.add(self.requests)
        self.event({"status": "request_reserved"})
        return self.requests

    def record_usage(self, index: int, usage: Mapping[str, Any]) -> None:
        require(index in self.unsettled_requests, "usage must settle one outstanding request")
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        require(
            all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys)
            and usage["prompt_tokens"] > 0
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "missing or inconsistent native reward-actor usage",
        )
        reserved = self.authority["budget"]["max_tokens_per_request"]
        require(usage["total_tokens"] <= reserved, "per-request token bound was violated")
        self.tokens += usage["total_tokens"] - reserved
        self.unsettled_requests.remove(index)
        self.event({"status": "usage_recorded", "request_index": index, "usage": dict(usage)})

    def reserve_update(self) -> None:
        self.check()
        require(
            self.updates < self.authority["budget"]["max_updates"], "update budget exhausted",
        )
        self.updates += 1
        self.event({"status": "update_reserved"})

    def require_example_capacity(self, chunks: int) -> None:
        self.check()
        budget = self.authority["budget"]
        require(
            type(chunks) is int and chunks > 0
            and self.classifier_calls + chunks <= budget["max_classifier_calls"]
            and self.requests < budget["max_reward_requests"]
            and self.tokens + budget["max_tokens_per_request"] <= budget["max_total_reward_tokens"]
            and self.updates < budget["max_updates"],
            "insufficient budget for every classifier chunk, one complete reward and one update",
        )

    def reserve_classifier_call(self, *, plan_sha256: str, chunk_index: int) -> None:
        self.check()
        require(
            self.classifier_calls < self.authority["budget"]["max_classifier_calls"],
            "classifier forward-call budget exhausted",
        )
        self.classifier_calls += 1
        self.event({
            "status": "classifier_forward_reserved", "chunk_index": chunk_index,
            "chunk_plan_sha256": plan_sha256,
        })


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any):
        raise ReadinessError("reward endpoint redirects are forbidden")


class FrozenQwenReward:
    """No retries, proxy inheritance, remote services or task-specific answer heuristics."""

    def __init__(
        self, lease: RemoteTrainingLease, *, official_scorer: Path,
        parse_answer: Callable[[str], str], parser_source: Path,
    ) -> None:
        lease.check()
        self.lease = lease
        authority = lease.authority
        checked_source(parser_source, authority["answer_parser_sha256"])
        require(
            Path(inspect.getfile(parse_answer)).resolve() == parser_source.resolve(),
            "answer parser must be the parent's frozen campaign parser, not an injected heuristic",
        )
        self.parse_answer = parse_answer
        source = checked_source(official_scorer, SCORER_SHA256)
        selected = definition_source(source, (
            "normalize_answer", "f1_score", "exact_match_score", "metric_max_over_ground_truths",
        ))
        self.metrics: dict[str, Any] = {"Counter": Counter, "re": re, "string": string}
        exec(compile(selected, str(official_scorer), "exec"), self.metrics)
        write_json(lease.root / "reward_provenance.json", {
            "identity": IDENTITY, "official_scorer_sha256": SCORER_SHA256,
            "official_scorer_revision": SCORER_REVISION,
            "scoring": "unchanged official per-example EM/F1; max over all gold aliases",
            "selected_definitions_sha256": digest(selected.encode()),
            "reward_metric": "official_f1_0_to_1",
            "answer_parser_sha256": authority["answer_parser_sha256"],
            "qwen_repository": authority["qwen_repository"],
            "qwen_revision": authority["qwen_revision"],
            "qwen_provenance_sha256": authority["qwen_provenance_sha256"],
            "qwen_frozen": True,
        })

    def score(
        self, messages: list[dict[str, str]], gold_aliases: Sequence[str],
    ) -> dict[str, Any]:
        require(
            isinstance(gold_aliases, (list, tuple)) and bool(gold_aliases)
            and all(isinstance(answer, str) for answer in gold_aliases),
            "all official gold aliases are required; never a selected alias",
        )
        lease = self.lease
        index = lease.reserve_request()
        authority = lease.authority
        payload = {
            **authority["decoding"], "model": authority["qwen_served_model"],
            "messages": messages, "stream": False,
        }
        write_json(lease.root / f"reward_{index:06d}.request.json", payload)
        try:
            request = Request(
                loopback_chat_url(authority["qwen_endpoint"]),
                data=json.dumps(payload).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"},
            )
            remaining = authority["budget"]["max_wall_seconds"] - (time.monotonic() - lease.started)
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=max(0.1, min(remaining, 120))) as response:
                raw = response.read(2 * 1024**2 + 1)
            (lease.root / f"reward_{index:06d}.response.json").write_bytes(raw)
            require(len(raw) <= 2 * 1024**2, "oversized reward response")
            data = json.loads(raw)
            lease.record_usage(index, data.get("usage", {}))
            require(data.get("model") == authority["qwen_served_model"], "served model identity changed")
            choices = data.get("choices", [])
            require(len(choices) == 1, "exactly one reward-actor output is required")
            content = choices[0].get("message", {}).get("content")
            require(isinstance(content, str), "reward actor did not return textual content")
            prediction = self.parse_answer(content)
            require(isinstance(prediction, str), "frozen answer parser must return a string")
            maximum = self.metrics["metric_max_over_ground_truths"]
            em = maximum(self.metrics["exact_match_score"], prediction, gold_aliases)
            f1 = maximum(self.metrics["f1_score"], prediction, gold_aliases)
            require(em in (0, 1) and math.isfinite(f1) and 0 <= f1 <= 1, "invalid official score")
            result = {
                "exact_match": int(em), "f1": float(f1), "prediction": prediction,
                "request_index": index, "usage": data["usage"],
            }
            write_json(lease.root / f"reward_{index:06d}.score.json", result)
            lease.check()
            return result
        except Exception as error:
            lease.fail(error)
            raise


class _UpdateLogger:
    def __init__(self, lease: RemoteTrainingLease) -> None:
        self.lease = lease

    def Train(self, values: Any) -> None:
        self.lease.event({"status": "author_policy_loss", "values": values})


class AuthorInstructionAdapter:
    """Training-only bridge, not a complete evaluator or baseline promotion API.

    The parent must load the pinned initializer into the patched author class,
    instantiate its single-process Accelerator, and validate training membership.
    There is deliberately no implicit checkpoint download or baseline fallback.
    """

    def __init__(
        self, *, lease: RemoteTrainingLease, author_root: Path, compressor: Any,
        accelerator: Any, reward: FrozenQwenReward,
    ) -> None:
        lease.check()
        require(reward.lease is lease, "reward and policy must share the acquisition budget")
        require(getattr(accelerator, "num_processes", None) == 1, "only single-process adaptation verified")
        original = checked_source(
            author_root / "experiments/taco-rl/train_reinforce.py",
            AUTHOR_HASHES["experiments/taco-rl/train_reinforce.py"],
        )
        core_path = author_root / "llmlingua/taco_rl/prompt_compressor_reinforce.py"
        patched = checked_source(core_path, PATCHED_SAMPLER_SHA256)
        require(
            patched.count(NEW_IMPORT) == patched.count(BOUND_DATASET_ARGUMENTS) == 1,
            "missing author import repair or explicit XLM-R dataset binding",
        )
        restored = patched.replace(NEW_IMPORT, OLD_IMPORT).replace(
            BOUND_DATASET_ARGUMENTS, UNBOUND_DATASET_ARGUMENTS,
        )
        require(
            digest(restored.encode()) == AUTHOR_HASHES[
                "llmlingua/taco-rl/prompt_compressor_reinforce.py"
            ] and type(compressor).__name__ == "PromptCompressorReinforce"
            and Path(inspect.getfile(type(compressor))).resolve() == core_path.resolve(),
            "only the hash-verified author sampler with import and XLM-R dataset repairs is supported",
        )
        for relative in ("llmlingua/prompt_compressor.py", "llmlingua/utils.py"):
            checked_source(author_root / relative, AUTHOR_HASHES[relative])
        require(compressor.max_seq_len == 512, "author sequence length must remain 512")
        initializer = lease.authority["initializer"]
        initializer_binding = verify_runtime_initializer_binding(lease, compressor)
        special_ids = validate_xlmr_special_tokens(compressor.tokenizer)
        import torch

        require(torch.is_grad_enabled(), "author sampling must retain autograd log probabilities")
        require(str(compressor.model.device).startswith("cuda"), "CPU/local fallback is forbidden")
        require(
            compressor.model.device == accelerator.device,
            "the author policy and Accelerator must use the same allocated CUDA device",
        )
        settings = lease.config["adaptation"]
        require(
            settings["batch_size"] == 1 and settings["policy_lr"] == 1e-5
            and settings["entropy_coeff"] == 0.01 and settings["compression_rate"] == 0.5
            and settings["compression_relaxation_tokens"] == 30,
            "the bounded author-default recipe changed; freeze a new adaptation protocol first",
        )
        # Author init resets the seed to 42; the declared seed must be applied after it.
        import random
        import numpy as np

        random.seed(lease.authority["seed"])
        np.random.seed(lease.authority["seed"])
        torch.manual_seed(lease.authority["seed"])
        torch.cuda.manual_seed_all(lease.authority["seed"])
        optimizer = torch.optim.AdamW(compressor.model.parameters(), lr=settings["policy_lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lease.authority["budget"]["max_updates"],
        )
        compressor.model, optimizer, scheduler = accelerator.prepare(
            compressor.model, optimizer, scheduler,
        )
        namespace = {
            "torch": torch, "accelerator": accelerator, "input_dict": {
                "entropy_coeff": settings["entropy_coeff"], "log_interval": 1,
            }, "csv_logger": _UpdateLogger(lease),
        }
        update_source = definition_source(original, ("REINFORCE",))
        require(digest(update_source.encode()) == AUTHOR_UPDATE_SHA256, "author update extraction changed")
        exec(compile(update_source, "verified-author-REINFORCE", "exec"), namespace)
        self.update = namespace["REINFORCE"](compressor.model, optimizer, scheduler)
        self.lease, self.compressor, self.reward = lease, compressor, reward
        self.torch = torch
        self.plan: InstructionChunkPlan | None = None
        self.plan_sha256: str | None = None
        self._active_chunk: int | None = None
        self._forward_handles = (
            compressor.model.register_forward_pre_hook(self._before_classifier_forward, with_kwargs=True),
            compressor.model.register_forward_hook(self._after_classifier_forward),
        )
        write_json(lease.root / "policy_provenance.json", {
            "identity": IDENTITY, "author_revision": AUTHOR_REVISION,
            "update_class_sha256": digest(update_source.encode()),
            "initializer_repository": INITIALIZER_REPOSITORY,
            "initializer_revision": INITIALIZER_REVISION,
            "trained_baseline": False, "model_training_mode": compressor.model.training,
            "sampler_changes": [
                "package rename", "AST-identical utility import repair",
                "TokenClfDataset constructor now receives model_name=self.model_name",
            ],
            "sampler_file_modified": True,
            "patched_sampler_sha256": PATCHED_SAMPLER_SHA256,
            "previous_import_only_sampler_sha256": IMPORT_ONLY_SAMPLER_SHA256,
            "categorical_sampling_statements_modified": False,
            "dataset_binding_patch_id": DATASET_BINDING_PATCH_ID,
            "xlmr_bos_eos_pad_ids": list(special_ids),
            "mutable_scope_id": MUTABLE_SCOPE_ID,
            "loss_changes": [], "seed_reset_after_author_init": lease.authority["seed"],
            "adapter_id": LONG_ADAPTER_ID, "source_patch_id": SOURCE_PATCH_ID,
            "initializer_receipt_sha256": initializer["receipt_sha256"],
            "initializer_files_manifest_sha256": initializer["files_manifest_sha256"],
            "initializer_byte_verification_owner": "parent; bound release attestation",
            "initializer_binding": initializer_binding,
            "long_instruction_update": "one full-prompt reward repeated for every chunk graph",
        })

    def _before_classifier_forward(self, module: Any, args: Any, kwargs: Any) -> None:
        require(self.plan is not None and self._active_chunk is not None, "unplanned classifier call")
        self.lease.reserve_classifier_call(
            plan_sha256=self.plan_sha256, chunk_index=self._active_chunk,
        )
        validate_xlmr_classifier_input(
            self.compressor.tokenizer, kwargs["input_ids"], kwargs["attention_mask"],
            author_tokens=self.plan.chunks[self._active_chunk].author_tokens,
        )

    def _after_classifier_forward(self, module: Any, args: Any, output: Any) -> None:
        self.lease.completed_classifier_calls += 1
        self.lease.event({
            "status": "classifier_forward_completed", "chunk_index": self._active_chunk,
            "chunk_plan_sha256": self.plan_sha256,
        })

    def prepare_instruction(self, envelope: InstructionEnvelope) -> dict[str, Any]:
        """Remote, tokenizer-only preparation; freeze the plan before any classifier call."""
        lease = self.lease
        lease.check()
        require(
            envelope.source_sha256 == lease.authority["instruction_source_sha256"],
            "instruction source is outside the training release",
        )
        if self.plan is not None:
            require(
                self.plan.system_sha256 == envelope.source_sha256
                and self.plan.source == envelope.instruction,
                "cannot replace the instruction plan within a training run",
            )
            return self.plan.receipt()

        def count(text: str) -> int:
            lease.check()
            return len(self.compressor.tokenizer.tokenize(text))

        plan = plan_instruction_chunks(
            envelope, count_tokens=count,
            tokenizer_manifest_sha256=lease.authority["initializer"]["tokenizer_manifest_sha256"],
        )
        required_calls = len(plan.chunks) * lease.authority["planned_total_updates"]
        require(
            required_calls <= lease.authority["budget"]["max_classifier_calls"],
            "classifier budget cannot cover the complete frozen training schedule",
        )
        receipt = plan.receipt()
        self.plan_sha256 = json_digest(receipt)
        write_json(lease.root / "instruction_mutable_span.json", envelope.mutable_span_receipt())
        write_json(lease.root / "instruction_chunk_plan.json", receipt)
        write_json(lease.root / "instruction_chunk_budget.json", {
            "chunk_plan_sha256": self.plan_sha256, "classifier_calls_planned": required_calls,
            "reward_calls_upper_bound": lease.authority["planned_total_updates"],
            "updates_planned": lease.authority["planned_total_updates"],
            "input_recomposition_is_exact": plan.recompose(plan.texts()) == envelope.instruction,
        })
        self.plan = plan
        lease.event({"status": "chunk_plan_frozen", "chunk_plan_sha256": self.plan_sha256})
        return receipt

    def train_one(
        self, envelope: InstructionEnvelope, gold_aliases: Sequence[str], *,
        training_split_manifest_sha256: str, example_id: str, epoch_index: int,
    ) -> dict[str, Any]:
        lease = self.lease
        lease.check()
        require(
            training_split_manifest_sha256 == lease.authority["training_split_manifest_sha256"]
            and isinstance(example_id, str) and bool(example_id),
            "parent must supply an ID from its frozen training-only split manifest",
        )
        require(
            envelope.source_sha256 == lease.config["instruction_source_sha256"][
                lease.authority["instruction_profile"]
            ],
            "instruction source is outside the frozen campaign",
        )
        require(
            type(epoch_index) is int
            and 0 <= epoch_index < lease.authority["planned_epochs"],
            "epoch index is outside the frozen training plan",
        )
        require(
            self.torch.is_grad_enabled()
            and lease.updates < lease.authority["budget"]["max_updates"],
            "autograd/update budget unavailable",
        )
        require(
            self.plan is not None and self.plan.source == envelope.instruction
            and self.plan.system_sha256 == envelope.source_sha256,
            "call prepare_instruction(envelope) and freeze its source-bound plan before training",
        )
        plan = self.plan
        lease.require_example_capacity(len(plan.chunks))
        classifier_before, reward_before, update_before = (
            lease.classifier_calls, lease.requests, lease.updates,
        )
        try:
            samples = []
            author_sample = getattr(self.compressor, "_PromptCompressorReinforce__compress")
            for index, (chunk, text) in enumerate(zip(plan.chunks, plan.texts())):
                lease.check()
                require(
                    len(self.compressor.tokenizer.tokenize(text)) == chunk.author_tokens,
                    "tokenizer no longer matches the frozen no-truncation chunk plan",
                )
                self._active_chunk = index
                before = lease.classifier_calls
                try:
                    output = author_sample(
                        [[text]], reduce_rate=0.5, token_to_word="mean", force_tokens=[],
                        token_map={}, force_reserve_digit=False, drop_consecutive=True,
                    )
                finally:
                    self._active_chunk = None
                require(
                    lease.classifier_calls == before + 1,
                    "one author classifier forward per planned chunk is required",
                )
                samples.append(author_chunk_sample(index, output))
            validate_chunk_samples(plan, samples)
            instruction = plan.recompose([sample.text for sample in samples])
            original = self.compressor.get_token_length(plan.source, use_oai_tokenizer=True)
            compressed = self.compressor.get_token_length(instruction, use_oai_tokenizer=True)
            require(
                type(original) is int and type(compressed) is int
                and original > 0 and compressed > 0,
                "empty author sample blocks rather than inventing a compression ratio",
            )
            messages = envelope.render(instruction)
            score = None
            if compressed > 0.1 * original:
                score = self.reward.score(messages, gold_aliases)
            value = compression_reward(
                original, compressed, None if score is None else score["f1"],
                rate=0.5, relaxation_tokens=30,
            )
            update_arguments = chunk_update_arguments(
                plan, samples, reward=value, compression_ratio=original / compressed,
            )
            lease.reserve_update()
            self.update.update(
                **update_arguments, epoch_counter=epoch_index, steps=lease.updates - 1,
            )
            lease.check()
            receipt = {
                "status": "author_update_completed", "example_id": example_id,
                "reward": value, "official_score": score, "original_instruction_tokens": original,
                "compressed_instruction_tokens": compressed,
                "source_system_sha256": envelope.source_sha256,
                "user_sha256": digest(envelope.user.encode()),
                "system_sha256": digest(messages[0]["content"].encode()),
                "adapter_id": LONG_ADAPTER_ID, "chunk_plan_sha256": self.plan_sha256,
                "mutable_source_span": envelope.mutable_span_receipt(),
                "chunk_count": len(plan.chunks),
                "classifier_calls": lease.classifier_calls - classifier_before,
                "reward_calls": lease.requests - reward_before,
                "policy_updates": lease.updates - update_before,
                "loss_shape": [len(plan.chunks), 512],
                "reward_broadcast": "same full-prompt scalar, not divided by chunk count",
                "public_benchmark_result": False, "checkpoint_ready_for_evaluation": False,
            }
            lease.event(receipt)
            return receipt
        except Exception as error:
            lease.fail(error)
            raise
