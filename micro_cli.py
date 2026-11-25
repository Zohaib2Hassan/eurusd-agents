"""Agent 3 - MicroEcon & Quant tutor.

MODE B - practice with automated grading:
  python micro_cli.py practice              random problem from the bank
  python micro_cli.py practice --problem 4  a bank problem
  python micro_cli.py practice --topic tax  a NEW generated problem (add --seed 7 to repeat one)
  python micro_cli.py quiz --n 3            several generated problems, one overall score
  python micro_cli.py quiz --n 4 --topics monopoly,cournot,stackelberg
  python micro_cli.py topics                generator topics
  python micro_cli.py list                  the fixed problem bank
  python micro_cli.py progress              your scores by topic + suggested next topic

MODE A - direct assistance (Given -> Formulas -> Steps -> Interpretation):
  python micro_cli.py solve market 500 10 -100 15 --floor 30
  python micro_cli.py solve monopoly 120 2 20 --fc 100
  python micro_cli.py solve cournot 180 1 30 30
  python micro_cli.py solve nash --row "3,0;5,1" --col "3,5;0,1" --names Coop,Defect
  python micro_cli.py solve consumer 0.5 100 2 1 --px-new 4
  python micro_cli.py solve cost 1 0.5 0.5 4 1 100
  python micro_cli.py solve expectancy 0.35 300 100 --n 100
  python micro_cli.py solve fx 50000 1.5 1.0850 1.0810 --target 1.0970 --winrate 0.45
  python micro_cli.py solve crypto 10000 1 90000 88200 --target 95400
  python micro_cli.py solve dca 1000 40000 32000 50000
  python micro_cli.py solve arb 40000 40800 --wfee 0.0005 --taker-sell 0.002 --fixed 10
  python micro_cli.py solve funding 60000 1 0.0004 90 0.0005 12000 30
  python micro_cli.py solve liq 3000 --lev 10 --mmr 0.005
  python micro_cli.py solve kelly 0.55 1 --fraction 0.5
  python micro_cli.py solve ruin 0.35 3 0.02 --dd 0.3
  python micro_cli.py solve cip 1.10 0.05 0.03 90

Market microstructure on real data:
  python micro_cli.py liquidity EURUSD                     (default 5-minute bars)
  python micro_cli.py liquidity BTCUSDT --source binance
"""
import argparse
import sys
from pathlib import Path

from microecon import econ, trading as tr
from microecon.bank import BANK
from microecon.generator import TOPICS, generate
from microecon.grader import run_interactive, tier
from microecon.progress import render as render_progress

ROOT = Path(__file__).parent


def solve(a):
    k = a.kind
    if k == "market":
        return econ.market(a.x[0], a.x[1], a.x[2], a.x[3], a.floor, a.ceiling, a.tax)[1]
    if k == "monopoly":
        return econ.monopoly(a.x[0], a.x[1], a.x[2], a.fc or 0)[1]
    if k == "cournot":
        return econ.cournot(a.x[0], a.x[1], a.x[2], a.x[3] if len(a.x) > 3 else a.x[2])[1]
    if k == "nash":
        parse = lambda s: [[float(v) for v in row.split(",")] for row in s.split(";")]
        names = a.names.split(",") if a.names else None
        return econ.nash_2player(parse(a.row), parse(a.col), names, names)[1]
    if k == "consumer":
        return econ.cobb_douglas_consumer(a.x[0], a.x[1], a.x[2], a.x[3], a.px_new)[1]
    if k == "cost":
        return econ.cobb_douglas_cost(*a.x[:6])[1]
    if k == "expectancy":
        return tr.expectancy(a.x[0], a.x[1], a.x[2], a.n)[1]
    if k == "fx":
        return tr.fx_position(a.x[0], a.x[1], a.x[2], a.x[3], a.symbol, a.target, a.winrate, a.quote_to_account)[1]
    if k == "crypto":
        return tr.crypto_units(a.x[0], a.x[1], a.x[2], a.x[3], a.target)[1]
    if k == "dca":
        return tr.dca(a.x[0], a.x[1:])[1]
    if k == "arb":
        return tr.spatial_arbitrage(a.x[0], a.x[1], a.qty, a.wfee, a.taker_buy, a.taker_sell, a.fixed)[1]
    if k == "funding":
        return tr.funding_carry(*a.x[:2], a.x[2], int(a.x[3]), a.x[4], a.x[5], a.x[6])[1]
    if k == "liq":
        return tr.liquidation(a.x[0], a.lev, a.mmr, a.side, a.margin, a.notional)[1]
    if k == "kelly":
        return tr.kelly(a.x[0], a.x[1], a.fraction)[1]
    if k == "ruin":
        return tr.risk_of_ruin(a.x[0], a.x[1], a.x[2], a.dd, int(a.n or 500))[1]
    if k == "cip":
        return tr.covered_interest_parity(a.x[0], a.x[1], a.x[2], int(a.x[3]))[1]
    raise SystemExit(f"unknown solver {k}")


def liquidity(a):
    import yaml
    from microecon.microstructure import liquidity_profile
    from quantx.data.sources import make_source
    from ta.validate import validate
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    src = make_source(a.source, cfg)
    if not src.connect():
        sys.exit(f"cannot connect to {a.source}")
    try:
        df, meta = src.fetch(a.symbol, a.tf, a.bars)
        df, warns, q, _ = validate(df, a.tf, calendar=meta["calendar"])
        point = pip = None
        if a.source == "mt5":
            info = src.mt5.symbol_info(a.symbol)
            point = info.point
            pip = point * 10 if info.digits in (3, 5) else point
        p = liquidity_profile(df, meta, point, pip)
    finally:
        src.close()
    print(f"## Liquidity profile: {a.symbol} ({a.tf}, {meta['source']}, {p['bars']} bars {p['start_utc'][:10]} -> {p['end_utc'][:10]})")
    if warns:
        print("Data warnings: " + "; ".join(warns))
    if a.tf in ("H4", "D1", "W1"):
        print("CAUTION: Roll and Corwin-Schultz were designed for daily EQUITY data with wide spreads. On liquid FX/crypto "
              "at this bar size they mostly measure price reversals and volatility, overstating the true spread by orders "
              "of magnitude. Use --tf M5 or M1 for a spread estimate.")
    for k, v in p.items():
        if isinstance(v, dict):
            print(f"- **{k}**: " + ", ".join(f"{kk} {vv}" for kk, vv in v.items()))
    print("\nEstimators: Roll (1984) J. Finance 39(4); Corwin & Schultz (2012) J. Finance 67(2); Amihud (2002) J. Financial Markets 5(1).")


def main():
    ap = argparse.ArgumentParser(description="Agent 3: MicroEcon & Quant tutor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("practice"); pr.add_argument("--problem", type=int)
    pr.add_argument("--topic", choices=sorted(TOPICS)); pr.add_argument("--seed", type=int)
    qz = sub.add_parser("quiz"); qz.add_argument("--n", type=int, default=3); qz.add_argument("--topics")
    qz.add_argument("--seed", type=int)
    sub.add_parser("list"); sub.add_parser("topics"); sub.add_parser("progress")
    so = sub.add_parser("solve")
    so.add_argument("kind")
    so.add_argument("x", nargs="*", type=float)
    for name, typ in (("--floor", float), ("--ceiling", float), ("--tax", float), ("--fc", float), ("--px-new", float),
                      ("--n", int), ("--target", float), ("--winrate", float), ("--quote-to-account", float),
                      ("--lev", float), ("--margin", float), ("--notional", float)):
        so.add_argument(name, type=typ)
    so.add_argument("--row"); so.add_argument("--col"); so.add_argument("--names")
    so.add_argument("--symbol", default="EURUSD")
    so.add_argument("--qty", type=float, default=1.0); so.add_argument("--wfee", type=float, default=0.0)
    so.add_argument("--taker-buy", type=float, default=0.0); so.add_argument("--taker-sell", type=float, default=0.0)
    so.add_argument("--fixed", type=float, default=0.0)
    so.add_argument("--mmr", type=float, default=0.005); so.add_argument("--side", default="long")
    so.add_argument("--fraction", type=float, default=0.5); so.add_argument("--dd", type=float, default=0.5)
    lq = sub.add_parser("liquidity"); lq.add_argument("symbol")
    lq.add_argument("--source", default="mt5", choices=["mt5", "binance"])
    lq.add_argument("--tf", default="M5"); lq.add_argument("--bars", type=int, default=3000)
    a = ap.parse_args()

    if a.cmd == "list":
        for p in BANK.values():
            corr = sum(1 for s in p.steps if s.correction)
            print(f"{p.pid}. {p.topic} ({p.difficulty}, {len(p.steps)} steps)" + (f"  [{corr} answer-key note]" if corr else ""))
    elif a.cmd == "topics":
        for k, (title, diff, _) in TOPICS.items():
            print(f"{k:12s} {title} ({diff})")
    elif a.cmd == "progress":
        print(render_progress())
    elif a.cmd == "practice":
        if a.topic:
            run_interactive(problem=generate(a.topic, a.seed))
        else:
            run_interactive(a.problem)
    elif a.cmd == "quiz":
        import random
        pool = a.topics.split(",") if a.topics else list(TOPICS)
        bad = [t for t in pool if t not in TOPICS]
        if bad:
            raise SystemExit(f"unknown topics {bad}; see: python micro_cli.py topics")
        rng = random.Random(a.seed)
        picks = [pool[i % len(pool)] for i in range(a.n)] if a.topics else rng.sample(pool, min(a.n, len(pool)))
        results = []
        for i, t in enumerate(picks, 1):
            print(f"\n======== Quiz problem {i}/{len(picks)} ========")
            sess = run_interactive(problem=generate(t, rng.randrange(10 ** 6)))
            results.append((sess.problem.topic, sum(sess.scores) * sess.weight))
            if sess.quit:
                break
        total = sum(sc for _, sc in results) / max(len(results), 1)
        print("\n## Quiz summary")
        for title, sc in results:
            print(f"- {title}: {sc:.1f}/100")
        print(f"**Overall: {total:.1f}% - {tier(total)}**")
    elif a.cmd == "solve":
        print(solve(a).render())
    elif a.cmd == "liquidity":
        liquidity(a)


if __name__ == "__main__":
    main()
