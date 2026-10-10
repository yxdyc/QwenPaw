#!/usr/bin/env python3
"""Executable toy checks for adaptive search statistics.

The numbers illustrate the contracts in tutorial_L2_adaptive_search.md. They
are not estimates of any production training or evaluation pipeline.
"""

from __future__ import annotations

import math
import random
import statistics


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def max_noise_scale(candidate_count: int, sigma: float = 1.0) -> float:
    return sigma * math.sqrt(2.0 * math.log(candidate_count))


def probability_any_above(candidate_count: int, z: float) -> float:
    return 1.0 - normal_cdf(z) ** candidate_count


def paired_z(loss_drops: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(loss_drops)
    standard_error = statistics.stdev(loss_drops) / math.sqrt(len(loss_drops))
    return mean, standard_error, mean / standard_error


def minimum_detectable_effect(
    sigma: float,
    runs_per_group: int,
    z_significance: float = 1.96,
    z_power: float = 0.84,
) -> float:
    return (z_significance + z_power) * math.sqrt(2.0 * sigma**2 / runs_per_group)


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    position_a = {item: index for index, item in enumerate(order_a)}
    position_b = {item: index for index, item in enumerate(order_b)}
    concordant = 0
    discordant = 0
    for left in range(len(order_a)):
        for right in range(left + 1, len(order_a)):
            x, y = order_a[left], order_a[right]
            same_direction = (position_a[x] - position_a[y]) * (
                position_b[x] - position_b[y]
            ) > 0
            concordant += int(same_direction)
            discordant += int(not same_direction)
    return (concordant - discordant) / (concordant + discordant)


def misrank_probability(margin: float, pairwise_sigma: float) -> float:
    return normal_cdf(-margin / pairwise_sigma)


def monte_carlo_maximum(candidate_count: int, trials: int = 20_000) -> float:
    rng = random.Random(7)
    maxima = [
        max(rng.gauss(0.0, 1.0) for _ in range(candidate_count))
        for _ in range(trials)
    ]
    return statistics.fmean(maxima)


def main() -> None:
    checks: list[bool] = []

    asymptotic = max_noise_scale(120)
    simulated = monte_carlo_maximum(120)
    any_three_sigma = probability_any_above(120, 3.0)
    print(
        "winner_noise",
        f"asymptotic={asymptotic:.3f}",
        f"monte_carlo_mean={simulated:.3f}",
        f"P(any>3sigma)={any_three_sigma:.3%}",
    )
    checks.extend([3.0 < asymptotic < 3.2, 2.0 < simulated < asymptotic, 0.14 < any_three_sigma < 0.17])

    mean, standard_error, z_score = paired_z([0.02, 0.08, 0.05])
    print("paired_loss", f"mean={mean:.3f}", f"SE={standard_error:.4f}", f"Z_L={z_score:.3f}")
    checks.extend([math.isclose(mean, 0.05), 2.8 < z_score < 3.0])

    mde = minimum_detectable_effect(sigma=0.5, runs_per_group=4)
    print("mde", f"value={mde:.3f}")
    checks.append(0.98 < mde < 1.00)

    tau = kendall_tau(["A", "B", "C", "D"], ["A", "C", "B", "D"])
    print("rank_stability", f"kendall_tau={tau:.3f}")
    checks.append(math.isclose(tau, 2.0 / 3.0))

    misrank = misrank_probability(margin=0.8, pairwise_sigma=0.4)
    print("pairwise_risk", f"P(mis-rank)={misrank:.3%}")
    checks.append(0.02 < misrank < 0.03)

    passed = sum(checks)
    print(f"SELF_CHECK {passed}/{len(checks)}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
