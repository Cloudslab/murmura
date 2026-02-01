"""Experiment 4: Validate FedRansel sampling probability conjecture (Eqs. 3-6).

This script validates Conjecture 1 from the FedRansel paper by generating
synthetic parameter vectors and verifying the probability equations:

Eq. 3: P(theta_j in S_i | r_i) = r_i
Eq. 4: P(theta_j, theta_j' in S_i | r_i) = r_i^2 for j != j' (large m)
Eq. 5: P(theta_j in S_i) = E[r_i] = (1 + T_l) / 2
Eq. 6: P(theta_j, theta_j' in S_i) = E[r_i^2] = (1 + T_l + T_l^2) / 3

Usage:
    python -m murmura.examples.fedransel.exp4_synthetic_validation
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ValidationResult:
    """Result from probability validation."""
    T_l: float
    m: int
    iterations: int

    # Marginal probability: P(theta_j in S_i)
    empirical_marginal: float
    theoretical_marginal: float
    marginal_error: float

    # Joint probability: P(theta_j, theta_j' in S_i)
    empirical_joint: float
    theoretical_joint: float
    joint_error: float


def validate_sampling_probabilities(
    m: int,
    T_l: float,
    num_iterations: int = 10000,
    seed: int = 42
) -> ValidationResult:
    """Validate sampling probability equations from Conjecture 1.

    Args:
        m: Number of parameters
        T_l: Minimum sampling threshold
        num_iterations: Number of sampling iterations for Monte Carlo estimation
        seed: Random seed

    Returns:
        ValidationResult with empirical vs theoretical probabilities
    """
    rng = np.random.default_rng(seed)

    # Track how many times each parameter index is sampled
    # We'll track indices 0, 1, and 2 as representatives
    sample_counts = np.zeros(3, dtype=np.int64)
    joint_counts = np.zeros(3, dtype=np.int64)  # (0,1), (0,2), (1,2)

    for _ in range(num_iterations):
        # Sample ratio r_i ~ U(T_l, 1)
        r_i = rng.uniform(T_l, 1.0)
        k_i = int(np.ceil(r_i * m))

        # Sample k_i indices without replacement
        sampled_indices = set(rng.choice(m, size=k_i, replace=False))

        # Track which of 0, 1, 2 are in the sampled set
        for idx in range(3):
            if idx in sampled_indices:
                sample_counts[idx] += 1

        # Track joint occurrences
        if 0 in sampled_indices and 1 in sampled_indices:
            joint_counts[0] += 1
        if 0 in sampled_indices and 2 in sampled_indices:
            joint_counts[1] += 1
        if 1 in sampled_indices and 2 in sampled_indices:
            joint_counts[2] += 1

    # Empirical probabilities
    empirical_marginal = float(np.mean(sample_counts)) / num_iterations
    empirical_joint = float(np.mean(joint_counts)) / num_iterations

    # Theoretical probabilities (Eqs. 5 and 6)
    theoretical_marginal = (1 + T_l) / 2
    theoretical_joint = (1 + T_l + T_l**2) / 3

    # Relative errors
    marginal_error = abs(empirical_marginal - theoretical_marginal) / theoretical_marginal
    joint_error = abs(empirical_joint - theoretical_joint) / theoretical_joint

    return ValidationResult(
        T_l=T_l,
        m=m,
        iterations=num_iterations,
        empirical_marginal=empirical_marginal,
        theoretical_marginal=theoretical_marginal,
        marginal_error=marginal_error,
        empirical_joint=empirical_joint,
        theoretical_joint=theoretical_joint,
        joint_error=joint_error,
    )


def validate_conditional_probabilities(
    m: int,
    T_l: float,
    num_iterations: int = 10000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Validate conditional probability equations (Eqs. 3 and 4).

    Args:
        m: Number of parameters
        T_l: Minimum sampling threshold
        num_iterations: Number of sampling iterations
        seed: Random seed

    Returns:
        Dictionary with results for different r_i values
    """
    rng = np.random.default_rng(seed)

    # Test at specific r_i values
    test_r_values = [T_l, (T_l + 1) / 2, 0.8, 1.0]
    results = {}

    for r_i in test_r_values:
        k_i = int(np.ceil(r_i * m))

        sample_count = 0
        joint_count = 0

        for _ in range(num_iterations):
            sampled_indices = set(rng.choice(m, size=k_i, replace=False))

            if 0 in sampled_indices:
                sample_count += 1
            if 0 in sampled_indices and 1 in sampled_indices:
                joint_count += 1

        empirical_conditional = sample_count / num_iterations
        empirical_joint_conditional = joint_count / num_iterations

        # Theoretical (Eqs. 3 and 4)
        theoretical_conditional = r_i
        theoretical_joint_conditional = r_i * r_i

        results[f"r={r_i:.2f}"] = {
            "empirical_P(j|r)": empirical_conditional,
            "theoretical_P(j|r)": theoretical_conditional,
            "error_P(j|r)": abs(empirical_conditional - theoretical_conditional) / theoretical_conditional if theoretical_conditional > 0 else 0,
            "empirical_P(j,j'|r)": empirical_joint_conditional,
            "theoretical_P(j,j'|r)": theoretical_joint_conditional,
            "error_P(j,j'|r)": abs(empirical_joint_conditional - theoretical_joint_conditional) / theoretical_joint_conditional if theoretical_joint_conditional > 0 else 0,
        }

    return results


def run_conjecture_validation(
    m_values: List[int] = None,
    T_l_values: List[float] = None,
    num_iterations: int = 10000,
    seed: int = 42
) -> List[ValidationResult]:
    """Run full conjecture validation across parameter grid.

    Args:
        m_values: List of parameter counts to test (default: [10^4, 10^5, 10^6])
        T_l_values: List of T_l thresholds to test (default: [0.3, 0.5, 0.7])
        num_iterations: Monte Carlo iterations per configuration
        seed: Random seed

    Returns:
        List of ValidationResult objects
    """
    if m_values is None:
        m_values = [10_000, 100_000, 1_000_000]
    if T_l_values is None:
        T_l_values = [0.3, 0.5, 0.7]

    results = []

    console.print("\n[bold]Experiment 4: Conjecture Validation[/bold]")
    console.print(f"Testing m in {m_values}, T_l in {T_l_values}")
    console.print(f"Iterations per config: {num_iterations}\n")

    for m in m_values:
        for T_l in T_l_values:
            console.print(f"Testing m={m:,}, T_l={T_l}...", end=" ")
            result = validate_sampling_probabilities(m, T_l, num_iterations, seed)
            results.append(result)
            console.print(f"marginal_err={result.marginal_error:.4%}, joint_err={result.joint_error:.4%}")

    return results


def display_results(results: List[ValidationResult]) -> None:
    """Display validation results in a formatted table."""
    table = Table(title="Conjecture 1 Validation Results")

    table.add_column("m", style="cyan")
    table.add_column("T_l", style="green")
    table.add_column("Emp P(j)", style="yellow")
    table.add_column("Theo P(j)", style="yellow")
    table.add_column("Error P(j)", style="red")
    table.add_column("Emp P(j,j')", style="yellow")
    table.add_column("Theo P(j,j')", style="yellow")
    table.add_column("Error P(j,j')", style="red")

    for r in results:
        table.add_row(
            f"{r.m:,}",
            f"{r.T_l:.1f}",
            f"{r.empirical_marginal:.4f}",
            f"{r.theoretical_marginal:.4f}",
            f"{r.marginal_error:.4%}",
            f"{r.empirical_joint:.4f}",
            f"{r.theoretical_joint:.4f}",
            f"{r.joint_error:.4%}",
        )

    console.print(table)

    # Summary
    avg_marginal_error = np.mean([r.marginal_error for r in results])
    avg_joint_error = np.mean([r.joint_error for r in results])

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Average marginal probability error: {avg_marginal_error:.4%}")
    console.print(f"  Average joint probability error: {avg_joint_error:.4%}")

    # Check if errors are within 1% as expected
    if avg_marginal_error < 0.01 and avg_joint_error < 0.01:
        console.print("  [green]Conjecture 1 validated: errors < 1% for large m[/green]")
    else:
        console.print("  [yellow]Warning: Some errors exceed 1%[/yellow]")


def main():
    """Run the full experiment 4 validation."""
    console.print("[bold blue]FedRansel Experiment 4: Sampling Probability Conjecture Validation[/bold blue]")
    console.print("=" * 70)

    # Run validation
    results = run_conjecture_validation(
        m_values=[10_000, 100_000, 1_000_000],
        T_l_values=[0.3, 0.5, 0.7],
        num_iterations=10_000,
        seed=42
    )

    # Display results
    display_results(results)

    # Also validate conditional probabilities for one configuration
    console.print("\n[bold]Conditional Probability Validation (m=100,000):[/bold]")
    cond_results = validate_conditional_probabilities(
        m=100_000,
        T_l=0.5,
        num_iterations=10_000,
        seed=42
    )

    cond_table = Table(title="Conditional Probabilities (Eqs. 3 & 4)")
    cond_table.add_column("r_i")
    cond_table.add_column("Emp P(j|r)")
    cond_table.add_column("Theo P(j|r)")
    cond_table.add_column("Error")
    cond_table.add_column("Emp P(j,j'|r)")
    cond_table.add_column("Theo P(j,j'|r)")
    cond_table.add_column("Error")

    for r_label, data in cond_results.items():
        emp_joint = data["empirical_P(j,j'|r)"]
        theo_joint = data["theoretical_P(j,j'|r)"]
        err_joint = data["error_P(j,j'|r)"]
        cond_table.add_row(
            r_label,
            f"{data['empirical_P(j|r)']:.4f}",
            f"{data['theoretical_P(j|r)']:.4f}",
            f"{data['error_P(j|r)']:.4%}",
            f"{emp_joint:.4f}",
            f"{theo_joint:.4f}",
            f"{err_joint:.4%}",
        )

    console.print(cond_table)


if __name__ == "__main__":
    main()
