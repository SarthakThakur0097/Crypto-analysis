# modules/fib_targets.py

from typing import List, Dict

def compute_extensions(
    base_low:  float,
    base_high: float,
    ratios:    List[float] = [1.0, 1.618]
) -> Dict[str, float]:
    """
    Given a base_low and base_high, return target levels:
      target_{r} = base_high + r * (base_high - base_low)
    for each r in ratios.
    """
    height = base_high - base_low
    return {
        f"target_{r}": base_high + r * height
        for r in ratios
    }
