#!/usr/bin/env python3
"""Debug script to check arithmetic operation return types."""

import torch
from logic_lang import RuleInterpreter


def debug_arithmetic_types():
    """Debug what types arithmetic operations return."""
    interpreter = RuleInterpreter()

    features = {
        "a": torch.tensor([[2.0]]),
        "b": torch.tensor([[3.0]]),
    }

    script = """
    expect a, b
    define sum_vars = a + b
    """

    interpreter.execute(script, features)

    result = interpreter.get_variable("sum_vars")
    print(f"Type: {type(result)}")
    print(f"Value: {result}")

    if hasattr(result, "value"):
        print(f"Has .value attribute: {result.value}")
        print(f"Value type: {type(result.value)}")


if __name__ == "__main__":
    debug_arithmetic_types()
