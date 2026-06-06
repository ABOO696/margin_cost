from margin_cost.calculator import MarginCostCalculator, MarginCostConfig
import argparse
from datetime import date, timedelta

from margin_cost.calculator import MarginCostCalculator, MarginCostConfig
from margin_cost.data import load_full_dataset, compute_longrun_avg_price
from margin_cost.indicators import (
    add_cost_deviation,
    add_institutional_status,
    add_short_deviation,
)
from margin_cost.institutional import InstitutionalCostCalculator, InstitutionalCostConfig
from margin_cost.interactive import build_interactive_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taiwan stock margin / institutional average cost analyser."
    )
    parser.add_argument(
        "--stock", default="2330",
        help="Stock ID (default: 2330)"
    )
    parser.add_argument(
        "--source", choices=["finmind", "goodinfo"], default="finmind",
        help="Data source: finmind (default) | goodinfo (web scraper, slower)",
    )
    parser.add_argument(
        "--start",
        default=(date.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date  YYYY-MM-DD  (default: 1 year ago)"
    )
    parser.add_argument(
        "--end",
        default=date.today().strftime("%Y-%m-%d"),
        help="End date  YYYY-MM-DD  (default: today)"
    )
    parser.add_argument(
        "--margin-method",
        choices=["moving_average", "back_calculate", "fifo_lifo_blend"],
        default="moving_average",
        help="Margin cost method: moving_average (default) | back_calculate | fifo_lifo_blend",
    )
    parser.add_argument(
        "--fifo-weight", type=float, default=0.80,
        help="FIFO weight for fifo_lifo_blend method (default 0.80 = 80%% FIFO + 20%% LIFO, backtest-optimised)",
    )
    parser.add_argument(
        "--inst-method", choices=["cumulative", "rolling"], default="cumulative",
        help="Trust / dealer cost method: cumulative | rolling (default: cumulative). Foreign always uses foreign_precise."
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Rolling window in trading days, only used when --method=rolling (default: 20)"
    )
    parser.add_argument(
        "--gap-anchor", type=float, default=None,
        help=(
            "Estimated average cost (TWD) for pre-history foreign positions. "
            "None = auto-compute 7-year volume-weighted average price (recommended)."
        ),
    )
    parser.add_argument(
        "--output", default="cost_{stock}.html",
        help="Output HTML filename (default: cost_{stock}.html)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.output.replace("{stock}", args.stock)

    print(f"Stock   : {args.stock}")
    print(f"Source  : {args.source}")
    print(f"Period  : {args.start}  ->  {args.end}")
    print(f"Margin  : {args.margin_method}")
    print(f"Foreign : foreign_precise (TSE-anchored hybrid)")
    print(f"Trust / Dealer: {args.inst_method}" + (f" (window={args.window})" if args.inst_method == "rolling" else ""))
    print(f"Output  : {output_file}")
    print("-" * 50)

    # 1. Load price, margin/short balance and institutional net buy/sell
    df = load_full_dataset(args.stock, args.start, args.end, source=args.source)

    # 2. Compute margin long / short average cost
    df = MarginCostCalculator(
        MarginCostConfig(method=args.margin_method, fifo_weight=args.fifo_weight)
    ).compute(df)

    # 3. Compute institutional average cost
    # Foreign investor: foreign_precise with recommended practical settings.
    #   gap_anchor_price     = 7-year VWAP (auto) or user-supplied --gap-anchor
    #   buy_price_adjustment = 0.995 (foreign tends to buy ~0.5% below market VWAP)
    #   lifo_decay_halflife_days = None (data coverage too low for decay to matter)
    if args.gap_anchor is not None:
        gap_anchor = args.gap_anchor
        print(f"Gap anchor : {gap_anchor:.2f} (user-supplied)")
    else:
        print("Computing 7-year VWAP as gap anchor price ...")
        gap_anchor = compute_longrun_avg_price(
            args.stock, years=7, end_date=args.end, source=args.source
        )
        print(f"Gap anchor : {gap_anchor:.2f} (7-year auto VWAP)")

    inst_cfg = InstitutionalCostConfig(
        method="foreign_precise",
        rolling_window=args.window,
        gap_anchor_price=gap_anchor,
        buy_price_adjustment=0.995,
        lifo_decay_halflife_days=None,
    )
    df = InstitutionalCostCalculator(inst_cfg).compute(df)

    # Re-run trust and dealer with the selected method if different
    if args.inst_method != "foreign_precise":
        from margin_cost.institutional import InstitutionalCostConfig as ICC
        td_cfg = ICC(
            method=args.inst_method,
            rolling_window=args.window,
            net_to_cost={"trust_net": "trust_cost", "dealer_net": "dealer_cost"},
        )
        df = InstitutionalCostCalculator(td_cfg).compute(df)

    # 4. Add deviation pct and profit/loss status for each group
    df = add_cost_deviation(df)
    df = add_short_deviation(df)
    df = add_institutional_status(df)

    # 5. Print the last 5 trading days
    cols = [
        "date", "close",
        "avg_margin_cost", "avg_short_cost",
        "foreign_cost", "trust_cost", "dealer_cost",
    ]
    if args.margin_method == "fifo_lifo_blend":
        cols += ["avg_margin_cost_fifo", "avg_margin_cost_lifo"]
    print(df[cols].tail().to_string(index=False))
    print("-" * 50)

    # 6. Export interactive HTML chart (open in any browser)
    build_interactive_chart(
        df,
        title=f"Stock {args.stock}  |  {args.start} ~ {args.end}  |  Margin / Institutional Average Cost",
        output_html=output_file,
    )
    print(f"Chart saved: {output_file}")


if __name__ == "__main__":
    main()
