"""Input-only routing for finite greedy coin-counting problems.

This is train-informed method engineering, not an automatically learned rule.
The gate parses public problem structure; it never accepts an ID, answer or score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

POLICY_VERSION = "aime-input-structure-counting/v1"
CANDIDATE_ID = "structure_gated_counting"


@dataclass(frozen=True)
class CountingStructure:
    denominations: tuple[int, ...]
    lower: int
    upper: int
    monetary_unit: str
    inventory: str = "unbounded"
    objective: str = "minimum_number_of_coins"
    algorithm: str = "largest_feasible_denomination_first"
    query: str = "count_integer_targets_where_greedy_is_optimal"

    def record(self):
        return asdict(self)


def counting_structure(question: str) -> CountingStructure | None:
    if not isinstance(question, str):
        raise TypeError("The applicability gate takes only public problem text")
    text = re.sub(r"\$([\d,]+)\$", r"\1", question).lower()
    coin_values = re.findall(
        r"\b(\d+)\s*[-–]?\s*(cents?|units?|dollars?|penn(?:y|ies)|pence)\s+coins?\b",
        text,
    )
    if not coin_values:
        return None
    normalize = {"cent": "cent", "cents": "cent", "unit": "unit", "units": "unit",
                 "dollar": "dollar", "dollars": "dollar", "penny": "penny",
                 "pennies": "penny", "pence": "penny"}
    units = {normalize[unit] for _, unit in coin_values}
    denominations = tuple(sorted({int(number) for number, _ in coin_values}))
    if len(units) != 1 or len(denominations) < 2 or denominations[0] != 1:
        return None
    unbounded = re.search(r"\b(?:unlimited|unbounded|unrestricted)\s+supply\b", text)
    greedy = re.search(r"\bgreedy\b", text)
    largest = re.search(r"\b(?:greatest|largest|highest)\s+(?:value|coin|denomination)\b", text)
    cardinality = re.search(
        r"\bfewer\s+coins\b|\bfewest\s+coins\b|\bminimum\s+(?:number|count)\s+of\s+coins\b",
        text,
    )
    counting = re.search(
        r"\bnumber\s+of\s+(?:values|amounts|targets|integers)\b|\bhow\s+many\s+(?:integer\s+)?(?:values|amounts|targets|integers)\b",
        text,
    )
    integer_domain = re.search(r"\bintegers?\b", text)
    limits = re.search(r"\bbetween\s+([\d,]+)\s+and\s+([\d,]+)\s+inclusive\b", text)
    if limits is None:
        limits = re.search(r"\bfrom\s+([\d,]+)\s+(?:through|to)\s+([\d,]+)\s+inclusive\b", text)
    if not all((unbounded, greedy, largest, cardinality, counting, integer_domain, limits)):
        return None
    lower, upper = (int(value.replace(",", "")) for value in limits.groups())
    if lower < 0 or upper < lower or any(value <= 0 for value in denominations):
        return None
    return CountingStructure(denominations, lower, upper, next(iter(units)))


def route(question: str):
    structure = counting_structure(question)
    return {
        "branch": "domain_guarded" if structure is not None else "no_skill",
        "structure": structure.record() if structure is not None else None,
        "inputs_used": ["question"],
        "reference_answer_used": False,
        "case_id_used": False,
    }


def minimum_count_table(denominations, maximum):
    """Exact analytic optimizer-side check; never called by routing or inference."""
    denominations = tuple(sorted(set(denominations)))
    if not denominations or denominations[0] != 1 or maximum < 0:
        raise ValueError("Analytical check requires unit denomination and a nonnegative bound")
    optimum = [0] * (maximum + 1)
    greedy = [0] * (maximum + 1)
    for target in range(1, maximum + 1):
        optimum[target] = 1 + min(optimum[target - coin] for coin in denominations if coin <= target)
        remainder = target
        for coin in reversed(denominations):
            count, remainder = divmod(remainder, coin)
            greedy[target] += count
        if remainder:
            raise AssertionError("Unit denomination must represent every target")
    return optimum, greedy
