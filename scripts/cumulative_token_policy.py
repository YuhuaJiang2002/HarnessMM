"""Explicit unlimited cumulative-token accounting for authorized continuations."""


def token_limit_allows(budget, field, prospective_tokens):
    mode = budget.get("token_usage_mode", "limited")
    ceiling = budget.get(field)
    if mode == "unlimited":
        if ceiling is not None:
            raise ValueError("Unlimited token accounting must use a null cumulative ceiling")
        return True
    if mode != "limited" or type(ceiling) is not int or ceiling <= 0:
        raise ValueError("Token accounting requires explicit unlimited mode or a positive legacy ceiling")
    return prospective_tokens <= ceiling
