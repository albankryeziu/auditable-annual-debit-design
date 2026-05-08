from __future__ import annotations

"""
build_results_pdf.py
====================

Build a compact results briefing PDF from the current benchmark schema.

Important discipline:
- This is a presentation layer only.
- It does not hardcode empirical conclusions.
- It derives all summaries directly from benchmark outputs.
- It is not the source of truth for the paper; the CSV/LaTeX tables remain the source of truth.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "pipeline_config.py").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline_config as cfg  # type: ignore

OUT_PATH = cfg.ROOT / "results_briefing.pdf"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bench = pd.read_csv(cfg.BENCH_PATH)
    bench["cost_M"] = pd.to_numeric(bench["cost_total"], errors="coerce") / 1e6
    bench["alignment_baseline"] = pd.to_numeric(bench.get("alignment_baseline", 0.0), errors="coerce")
    bench["budget_util"] = pd.to_numeric(bench.get("budget_util", 0.0), errors="coerce")
    bench["slack_total"] = pd.to_numeric(bench.get("slack_total", 0.0), errors="coerce")
    bench["year"] = pd.to_numeric(bench["year"], errors="coerce").astype("Int64")

    log = pd.read_csv(cfg.M3_LOG_PATH) if cfg.M3_LOG_PATH.exists() else pd.DataFrame()
    kkt = pd.read_csv(cfg.KKT_PATH) if cfg.KKT_PATH.exists() else pd.DataFrame()
    return bench, log, kkt


def _main_benchmark(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[bench["rule"].isin(["M1", "M2", "M3"])].copy().sort_values(["rule", "year"])


def _feature_ladder(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[(bench["rule"] == "M3_feature_ladder") & (bench["variant"] == cfg.MAIN_M3_VARIANT)].copy().sort_values(["feature_set", "year"])


def _variant_compare(bench: pd.DataFrame) -> pd.DataFrame:
    return bench[(bench["rule"] == "M3_variant_comparison") & (bench["feature_set"] == cfg.MAIN_FEATURE_SET)].copy().sort_values(["variant", "year"])


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _table(data: List[List[str]], col_widths: List[float]) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2166AC")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _figure(path: Path, width: float, height: float) -> Image | None:
    if not path.exists():
        return None
    return Image(str(path), width=width * inch, height=height * inch)


def _summary_text(bench: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    main = _main_benchmark(bench)
    holdout_year = cfg.EVAL_YEARS[-1]
    holdout = main[main["year"] == holdout_year].set_index("rule")
    if {"M1", "M2", "M3"}.issubset(set(holdout.index)):
        m1 = float(holdout.loc["M1", "cost_M"])
        m2 = float(holdout.loc["M2", "cost_M"])
        m3 = float(holdout.loc["M3", "cost_M"])
        d12 = (m1 - m2) / max(abs(m2), 1e-12) * 100.0
        d23 = (m2 - m3) / max(abs(m2), 1e-12) * 100.0
        lines.append(
            f"Holdout year {holdout_year}: M1 cost = {m1:.2f}M EUR, M2 cost = {m2:.2f}M EUR, and main M3 cost = {m3:.2f}M EUR. Relative to M2, M1 changes cost by {d12:+.1f}% and M3 changes cost by {-d23:+.1f}% when expressed as (cost_M3 - cost_M2)/cost_M2."
        )
    ladder = _feature_ladder(bench)
    sub = ladder[ladder["year"] == holdout_year]
    if not sub.empty:
        best = sub.loc[sub["cost_M"].idxmin()]
        lines.append(
            f"Within the feature ladder at holdout {holdout_year}, the lowest-cost feature set is {best['feature_set']} under variant {best['variant']} with cost {best['cost_M']:.2f}M EUR."
        )
    variants = _variant_compare(bench)
    if not variants.empty:
        summary = variants.groupby("variant", as_index=False)["cost_M"].mean().sort_values("cost_M")
        best_var = summary.iloc[0]
        lines.append(
            f"Across evaluation years at feature set {cfg.MAIN_FEATURE_SET}, the lowest mean-cost M3 architecture is {best_var['variant']} with mean cost {best_var['cost_M']:.2f}M EUR."
        )
    return lines


def build_pdf() -> None:
    if not cfg.BENCH_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark results: {cfg.BENCH_PATH}")

    bench, log, kkt = _load()

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=17, alignment=TA_CENTER, textColor=colors.HexColor("#1a3a5c"), spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, textColor=colors.HexColor("#1a3a5c"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10.5, textColor=colors.HexColor("#2166AC"), spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.8, leading=12, alignment=TA_JUSTIFY, spaceAfter=6)
    cap = ParagraphStyle("cap", parent=styles["Normal"], fontSize=7.8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=8)

    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(_paragraph("Results briefing", title))
    story.append(_paragraph("Auditable Annual Debit Design for Non-Firm Curtailment Rights under Partial Observability", h2))
    story.append(Spacer(1, 8))
    story.append(_paragraph(
        "This briefing is generated directly from the benchmark outputs in the final Windows-native repo. It is a presentation artifact only: the benchmark CSV files and LaTeX tables remain the source of truth.",
        body,
    ))

    story.append(_paragraph("1. Main benchmark", h1))
    main = _main_benchmark(bench)
    piv = main.pivot_table(index="rule", columns="year", values="cost_M", aggfunc="first").reindex(["M1", "M2", "M3"])
    header = ["Rule", *[str(y) for y in cfg.EVAL_YEARS], "Mean"]
    rows = [header]
    for rule in ["M1", "M2", "M3"]:
        row = [rule]
        for y in cfg.EVAL_YEARS:
            val = piv.loc[rule, y] if y in piv.columns else float("nan")
            row.append("—" if pd.isna(val) else f"{float(val):.2f}")
        row.append(f"{float(piv.loc[rule].mean()):.2f}")
        rows.append(row)
    story.append(_table(rows, [1.0 * inch] + [0.75 * inch] * len(cfg.EVAL_YEARS) + [0.8 * inch]))
    story.append(Spacer(1, 6))
    fig = _figure(cfg.FIGURES_DIR / "F1_main_benchmark.png", 6.2, 2.8)
    if fig is not None:
        story.append(fig)
        story.append(_paragraph("Figure 1. Main benchmark generated from the current benchmark schema.", cap))

    for line in _summary_text(bench):
        story.append(_paragraph(line, body))

    story.append(_paragraph("2. Feature ladder and M3 architecture comparison", h1))
    ladder = _feature_ladder(bench)
    if not ladder.empty:
        holdout = cfg.EVAL_YEARS[-1]
        sub = ladder[ladder["year"] == holdout].copy().sort_values("feature_set")
        rows = [["Feature set", "Cost (M)", "Alignment", "Budget util.", "Slack"]]
        for _, r in sub.iterrows():
            rows.append([
                str(r["feature_set"]),
                f"{float(r['cost_M']):.2f}",
                f"{float(r['alignment_baseline']):.1f}",
                f"{float(r['budget_util']):.3f}",
                f"{float(r['slack_total']):.1f}",
            ])
        story.append(_table(rows, [1.0 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch]))
        story.append(Spacer(1, 6))
        fig2 = _figure(cfg.FIGURES_DIR / "F2_feature_ladder.png", 5.4, 2.6)
        if fig2 is not None:
            story.append(fig2)
            story.append(_paragraph("Figure 2. Feature-ladder comparison for the holdout year.", cap))

    variants = _variant_compare(bench)
    if not variants.empty:
        rows = [["Variant", *[str(y) for y in cfg.EVAL_YEARS], "Mean"]]
        piv_v = variants.pivot_table(index="variant", columns="year", values="cost_M", aggfunc="first").reindex(cfg.M3_VARIANTS)
        for variant in cfg.M3_VARIANTS:
            row = [variant]
            vals = []
            for y in cfg.EVAL_YEARS:
                val = piv_v.loc[variant, y] if y in piv_v.columns else float("nan")
                row.append("—" if pd.isna(val) else f"{float(val):.2f}")
                vals.append(val)
            row.append(f"{float(pd.Series(vals, dtype=float).mean()):.2f}")
            rows.append(row)
        story.append(_table(rows, [1.1 * inch] + [0.75 * inch] * len(cfg.EVAL_YEARS) + [0.8 * inch]))
        story.append(Spacer(1, 6))
        fig3 = _figure(cfg.FIGURES_DIR / "F3_m3_variants.png", 6.2, 2.7)
        if fig3 is not None:
            story.append(fig3)
            story.append(_paragraph("Figure 3. M3 architecture comparison at the main feature set.", cap))

    story.append(PageBreak())
    story.append(_paragraph("3. Training and diagnostics", h1))
    if not log.empty:
        log_small = log.sort_values(["feature_set", "variant", "iter"]).head(20)
        rows = [["Feature", "Variant", "Iter.", "Δθ_rel", "Train cost", "ᾱ"]]
        for _, r in log_small.iterrows():
            rows.append([
                str(r.get("feature_set", "")),
                str(r.get("variant", "")),
                str(int(r.get("iter", 0))),
                "—" if pd.isna(r.get("delta_theta_rel", pd.NA)) else f"{float(r['delta_theta_rel']):.5f}",
                "—" if pd.isna(r.get("train_cost", pd.NA)) else f"{float(r['train_cost']):.2f}",
                "—" if pd.isna(r.get("alpha_mean", pd.NA)) else f"{float(r['alpha_mean']):.3f}",
            ])
        story.append(_table(rows, [0.7 * inch, 1.0 * inch, 0.45 * inch, 0.8 * inch, 0.9 * inch, 0.6 * inch]))
        story.append(Spacer(1, 6))
        fig4 = _figure(cfg.FIGURES_DIR / "F4_training_convergence.png", 6.2, 2.7)
        if fig4 is not None:
            story.append(fig4)
            story.append(_paragraph("Figure 4. Training convergence diagnostics.", cap))

    if not kkt.empty:
        story.append(_paragraph("KKT diagnostics", h2))
        kkt = kkt.copy()
        kkt["equalization_error"] = pd.to_numeric(kkt.get("equalization_error", 0.0), errors="coerce")
        summary = (
            kkt[kkt["equalization_error"].notna()]
            .groupby(["rule", "year"], as_index=False)
            .agg(mean_equalization_error=("equalization_error", "mean"), max_equalization_error=("equalization_error", "max"))
            .sort_values(["rule", "year"])
        )
        rows = [["Rule", "Year", "Mean err.", "Max err."]]
        for _, r in summary.head(20).iterrows():
            rows.append([
                str(r["rule"]),
                str(int(r["year"])),
                f"{float(r['mean_equalization_error']):.4f}",
                f"{float(r['max_equalization_error']):.4f}",
            ])
        story.append(_table(rows, [1.1 * inch, 0.6 * inch, 0.9 * inch, 0.9 * inch]))

    fig5 = _figure(cfg.FIGURES_DIR / "F5_cost_vs_alignment.png", 5.4, 2.8)
    if fig5 is not None:
        story.append(Spacer(1, 8))
        story.append(fig5)
        story.append(_paragraph("Figure 5. Cost-versus-alignment plot for the holdout year.", cap))

    doc.build(story)


def main() -> None:
    print("Building results briefing PDF …")
    build_pdf()
    print(f"✓ {OUT_PATH}")


if __name__ == "__main__":
    main()
