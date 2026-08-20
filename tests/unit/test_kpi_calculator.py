from __future__ import annotations

import pytest

from app.tools.kpi_calculator import KpiCalculationError, calculate_kpi_projection


def test_calculate_kpi_projection_basic_math():
    result = calculate_kpi_projection(
        budget=1000.0, assumed_cost_per_click=2.0, assumed_conversion_rate=0.05
    )
    assert result.projected_clicks == 500
    assert result.projected_conversions == 25
    assert result.projected_cost_per_acquisition == pytest.approx(40.0)
    assert result.is_estimate is True


def test_calculate_kpi_projection_zero_conversions_reports_negative_cpa_sentinel():
    result = calculate_kpi_projection(
        budget=1.0, assumed_cost_per_click=1.0, assumed_conversion_rate=0.001
    )
    assert result.projected_conversions == 0
    assert result.projected_cost_per_acquisition == -1


@pytest.mark.parametrize(
    "budget,cpc,rate",
    [(-1, 1.0, 0.1), (0, 1.0, 0.1), (100, -1, 0.1), (100, 0, 0.1), (100, 1.0, 0), (100, 1.0, 1.5)],
)
def test_calculate_kpi_projection_rejects_invalid_inputs(budget, cpc, rate):
    with pytest.raises(KpiCalculationError):
        calculate_kpi_projection(budget=budget, assumed_cost_per_click=cpc, assumed_conversion_rate=rate)
