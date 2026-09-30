"""Ad hoc (human request, 2026-09-30): exp04's F1 survival curves, baseline and stacked prompt overlaid.

Per (model, rule): each prompt-dependent arm (no prefill, compliant prefill, non-compliant prefill) under the baseline
prompt (solid, thick) and the stacked prompt (dashed, thin), same color per arm. The no-rule prefill and the
thinking-off ceiling do not depend on the prompt (identical in both F1s) and are left out. Scoring as the 'main' run
(an empty thinking trace is censored at token 0).

Output (never overwritten): figures/exp04_prefill/adhoc_f1_overlay/F1c_survival_baseline_vs_stacked.{html,png}
Run: /venv/main/bin/python scripts/adhoc_exp04_f1_overlay.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cc_config as cfg
import cc_exp04_analysis as a4
from cc_survival import kaplan_meier

RUN = "adhoc_f1_overlay"
CONDITIONS = ["none", "prefill_compliant", "prefill_noncompliant"]
PROMPT_LINE = {"baseline": {"dash": "solid", "width": 2.6}, "stacked": {"dash": "dash", "width": 1.6}}


def main() -> None:
    fig_dir = cfg.EXP04.figure_dir(RUN)
    if fig_dir.exists():
        raise SystemExit(f"{fig_dir} exists; analysis runs are never overwritten")
    df = a4.analysis_rows(a4.load())
    models = a4.models_in(df)
    grid = np.arange(0, 2001, 10, dtype=float)
    fig = make_subplots(rows=len(models), cols=len(a4.RULES), shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[a4.MODE_LABEL[m] for m in a4.RULES] + [""] * (len(models) - 1) * len(a4.RULES),
                        vertical_spacing=0.03, horizontal_spacing=0.02)
    for i, model in enumerate(models):
        for j, mode in enumerate(a4.RULES):
            for condition in CONDITIONS:
                for prompt, line in PROMPT_LINE.items():
                    cell = a4.arm(df, model, condition, prompt)
                    cell = cell[cell["mode"] == mode]
                    if cell.empty:
                        continue
                    x = grid[grid <= cell["time"].max()]  # KM is defined up to the longest observed trace
                    s = 100 * kaplan_meier(cell["time"].to_numpy(), cell["event"].to_numpy(),
                                           np.ones((1, len(cell))), x)[0]
                    label = f"{cfg.CONDITION_STYLE[condition]['label']}, {prompt} prompt"
                    fig.add_trace(go.Scatter(x=x, y=s, mode="lines", name=label, legendgroup=label,
                                             showlegend=i == 0 and j == 0,
                                             line={"color": cfg.CONDITION_STYLE[condition]["color"], **line},
                                             hovertemplate=f"{model} | {mode} | {label}<br>t=%{{x}}: "
                                                           "%{y:.0f}%<extra></extra>"),
                                  row=i + 1, col=j + 1)
            fig.add_vline(x=a4.T_STAR, line={"color": "black", "width": 0.5, "dash": "dot"}, row=i + 1, col=j + 1)
        fig.update_yaxes(title_text=f"<b>{model}</b><br>% no violation yet", row=i + 1, col=1)
    fig.update_xaxes(title_text="tokens into the graded text", row=len(models))
    fig.update_xaxes(range=[0, grid.max()])
    fig.update_yaxes(range=[0, 101])
    height = 230 * len(models) + 280
    fig.update_layout(title="F1c. Baseline vs stacked prompt: share of traces with no rule violation yet (KM)<br>"
                            "<sup><b>solid thick = baseline prompt, dashed thin = stacked prompt</b>; color = arm. "
                            "Prefill rows are scored from the first token after the prefill; dotted line: t* = "
                            f"{a4.T_STAR}; each curve stops at its longest trace</sup>",
                      legend={"orientation": "h", "y": -110 / (height - 280), "yanchor": "top", "x": 0.5,
                              "xanchor": "center"},
                      margin={"t": 120, "b": 150})
    fig_dir.mkdir(parents=True)
    print(a4.save(fig, fig_dir, "F1c_survival_baseline_vs_stacked", 1500, height))


if __name__ == "__main__":
    main()
