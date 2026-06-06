"""Interactive Plotly chart with confidence bands, range selector and range slider."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COST_SERIES: dict[str, tuple[str, str]] = {
    "avg_margin_cost": ("Margin Long Cost",  "#d62728"),
    "avg_short_cost":  ("Margin Short Cost", "#2ca02c"),
    "foreign_cost":    ("Foreign Cost",      "#1f77b4"),
    "trust_cost":      ("Trust Cost",        "#ff7f0e"),
    "dealer_cost":     ("Dealer Cost",       "#9467bd"),
}

_FLOW_SERIES: dict[str, tuple[str, str]] = {
    "foreign_net": ("Foreign Net", "#1f77b4"),
    "trust_net":   ("Trust Net",   "#ff7f0e"),
    "dealer_net":  ("Dealer Net",  "#9467bd"),
}

_RANGE_BUTTONS = [
    dict(count=1,  label="1M",  step="month",  stepmode="backward"),
    dict(count=3,  label="3M",  step="month",  stepmode="backward"),
    dict(count=6,  label="6M",  step="month",  stepmode="backward"),
    dict(count=1,  label="YTD", step="year",   stepmode="todate"),
    dict(count=1,  label="1Y",  step="year",   stepmode="backward"),
    dict(step="all", label="All"),
]


def build_interactive_chart(
    df: pd.DataFrame,
    title: str = "Institutional-Grade Margin / Short / Institutional Average Cost",
    date_col: str = "date",
    close_col: str = "close",
    output_html: str | None = "margin_cost.html",
) -> go.Figure:
    """Two-panel interactive chart with confidence bands.

    Upper panel : close price + cost lines + confidence band shading.
    Lower panel : institutional net buy/sell bars.
    Range selector buttons (1M/3M/6M/YTD/1Y/All) and drag range slider included.
    Only columns present in df are drawn.
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price and Average Cost (with Confidence Band)",
                        "Institutional Net Buy/Sell"),
    )

    # Close price
    fig.add_trace(
        go.Scatter(x=data[date_col], y=data[close_col],
                   name="Close", line=dict(color="#222222", width=2)),
        row=1, col=1,
    )

    # Cost lines + confidence band
    for cost_col, (name, color) in _COST_SERIES.items():
        if cost_col not in data.columns:
            continue

        upper_col = f"{cost_col}_upper"
        lower_col = f"{cost_col}_lower"
        has_band  = upper_col in data.columns and lower_col in data.columns

        # Confidence band (filled area between upper and lower)
        if has_band:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([data[date_col], data[date_col][::-1]]),
                    y=pd.concat([data[upper_col], data[lower_col][::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.08,
                    line=dict(width=0),
                    name=f"{name} Band",
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
            # Upper / lower edge lines (thin, same color)
            for edge_col, dash in [(upper_col, "dot"), (lower_col, "dot")]:
                fig.add_trace(
                    go.Scatter(x=data[date_col], y=data[edge_col],
                               line=dict(color=color, width=0.8, dash=dash),
                               showlegend=False, hoverinfo="skip"),
                    row=1, col=1,
                )

        # Centre cost line
        fig.add_trace(
            go.Scatter(x=data[date_col], y=data[cost_col],
                       name=name,
                       line=dict(color=color, width=1.6, dash="dash")),
            row=1, col=1,
        )

    # Net buy/sell bars
    for flow_col, (name, color) in _FLOW_SERIES.items():
        if flow_col in data.columns:
            fig.add_trace(
                go.Bar(x=data[date_col], y=data[flow_col],
                       name=name, marker_color=color, opacity=0.6),
                row=2, col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03,
                    xanchor="left", x=0),
        bargap=0.1,
        height=860,
        xaxis=dict(
            rangeselector=dict(
                buttons=_RANGE_BUTTONS,
                bgcolor="#f0f0f0",
                activecolor="#aec7e8",
            ),
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
    )
    fig.update_yaxes(title_text="Price (TWD)", row=1, col=1)
    fig.update_yaxes(title_text="Shares",      row=2, col=1)

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn")

    return fig
