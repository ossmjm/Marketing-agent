"""
Campaign KPI calculator tool.

Why this tool exists and when it's used
----------------------------------------
The assignment asks for at most one well-justified tool. LLMs are
unreliable at arithmetic and at silently "estimating" numbers that
sound plausible but are made up. A KPI calculator gives the agent a
deterministic way to turn budget + assumptions into projected reach /
cost-per-acquisition numbers *without* asking the LLM to invent them.

The PLANNER decides this tool is needed only when the user's query
contains an explicit budget figure (e.g. "$5,000 budget") -- see
`agent.py`. This keeps tool invocation intentional rather than
"always call every tool just in case."

Important honesty constraint: this tool does NOT know real-world
advertising benchmarks. It requires the caller to supply an assumed
cost-per-click / conversion-rate, and it labels its own output as a
projection based on the given assumptions -- never as a guaranteed
outcome. If no assumption is available from the KB or the user, the
agent should not call this tool (see `AbstentionResponse` path).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KpiProjection:
    budget: float
    assumed_cost_per_click: float
    assumed_conversion_rate: float
    projected_clicks: int
    projected_conversions: int
    projected_cost_per_acquisition: float
    is_estimate: bool = True


class KpiCalculationError(ValueError):
    pass


def calculate_kpi_projection(
    budget: float,
    assumed_cost_per_click: float,
    assumed_conversion_rate: float,
) -> KpiProjection:
    if budget <= 0:
        raise KpiCalculationError("budget must be positive")
    if assumed_cost_per_click <= 0:
        raise KpiCalculationError("assumed_cost_per_click must be positive")
    if not (0 < assumed_conversion_rate <= 1):
        raise KpiCalculationError("assumed_conversion_rate must be in (0, 1]")

    projected_clicks = int(budget / assumed_cost_per_click)
    projected_conversions = int(projected_clicks * assumed_conversion_rate)
    cpa = (budget / projected_conversions) if projected_conversions > 0 else float("inf")

    return KpiProjection(
        budget=budget,
        assumed_cost_per_click=assumed_cost_per_click,
        assumed_conversion_rate=assumed_conversion_rate,
        projected_clicks=projected_clicks,
        projected_conversions=projected_conversions,
        projected_cost_per_acquisition=round(cpa, 2) if cpa != float("inf") else -1,
    )
