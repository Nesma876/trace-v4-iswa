#!/usr/bin/env python3
"""Generate the five TRACE ISWA figures from frozen, adjudicated V4 tables only."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
MODELS = ["linear_ar", "gru", "transformer", "rssm"]
LABEL = {"linear_ar": "Linear AR", "gru": "GRU", "transformer": "Transformer", "rssm": "RSSM"}
COLOR = {"linear_ar": "#64748b", "gru": "#0f766e", "transformer": "#2563eb", "rssm": "#c2410c"}
HORIZONS = [4, 12, 24, 48]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 160,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2)); ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.16, 0.25, "PhysioNet 2019\nSet A", "20,336 patients"),
        (0.23, 0.58, 0.18, 0.25, "Train / validation", "14,198 / 3,011\nselection by 1-step NLL"),
        (0.46, 0.58, 0.18, 0.25, "Locked Set A test", "3,127 patients\ncommon 48 h support"),
        (0.69, 0.58, 0.18, 0.25, "Held-out Set B", "20,000 patients\nunchanged evaluation"),
        (0.25, 0.10, 0.50, 0.25, "TRACE evaluation axes", "Predictive fit | recursive state | temporal change\nconditional intervals | observation density"),
    ]
    for x, y, w, h, title, sub in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", lw=1.2,
                               ec="#334155", fc="#f8fafc")
        ax.add_patch(patch); ax.text(x+w/2, y+h*0.66, title, ha="center", va="center", weight="bold")
        ax.text(x+w/2, y+h*0.29, sub, ha="center", va="center", color="#475569", fontsize=8)
    for a, b in [((0.18, .705), (.23, .705)), ((.41, .705), (.46, .705)), ((.64, .705), (.69, .705))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
    ax.annotate("", xy=(.5, .35), xytext=(.55, .58), arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
    ax.text(.5, .96, "TRACE evaluation protocol", ha="center", va="top", fontsize=14, weight="bold")
    ax.text(.5, .015, "No treatment-action channel; observational trajectory evaluation only", ha="center", color="#64748b")
    save(fig, "pipeline.pdf")


def state_panel(ax, table: Path, title: str) -> None:
    data = rows(table)
    for model in MODELS:
        ys=[]; lo=[]; hi=[]
        for h in HORIZONS:
            r=next(x for x in data if x["model"]==model and x["dimension"]==f"{h}h_state")
            y=float(r["mean"]); ys.append(y); lo.append(y-float(r["ci_low"])); hi.append(float(r["ci_high"])-y)
        ax.errorbar(HORIZONS, ys, yerr=np.array([lo,hi]), marker="o", capsize=2.5,
                    lw=1.5, color=COLOR[model], label=LABEL[model])
    ax.set(title=title, xlabel="Forecast horizon (hours)", ylabel="Normalized MacroMAE", xticks=HORIZONS)
    ax.grid(alpha=.18); ax.legend(frameon=False)


def figure2() -> None:
    fig, axes=plt.subplots(1,2,figsize=(10.5,4.1),sharex=True)
    state_panel(axes[0], ROOT/"outputs/v4_internal/tables/bootstrap_intervals.csv", "Locked Set A")
    state_panel(axes[1], ROOT/"outputs/v4_external/tables/bootstrap_intervals.csv", "Held-out Set B")
    fig.suptitle("Recursive rollout fidelity", weight="bold"); fig.tight_layout()
    save(fig,"figure2_rollout_fidelity.pdf")


def dynamic_panel(ax, table: Path, title: str) -> None:
    data=rows(table)
    for model in MODELS:
        rs=[next(x for x in data if x["model"]==model and x["cohort"]=="common_48h" and int(x["horizon"])==h) for h in HORIZONS]
        ax.plot(HORIZONS,[float(r["dynamic_mae_mean"]) for r in rs],marker="o",lw=1.5,color=COLOR[model],label=LABEL[model])
    ax.set(title=title,xlabel="Forecast horizon (hours)",ylabel="DynamicMAE",xticks=HORIZONS); ax.grid(alpha=.18)


def figure3() -> None:
    fig=plt.figure(figsize=(10.5,7)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.05])
    a,b,c=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])
    dynamic_panel(a,ROOT/"outputs/v4_internal/tables/table_3_dynamic_fidelity.csv","Locked Set A")
    dynamic_panel(b,ROOT/"outputs/v4_external/tables/table_3_dynamic_fidelity.csv","Held-out Set B"); b.legend(frameon=False)
    width=.36; x=np.arange(len(MODELS))
    for offset,(scope,path) in zip([-.18,.18],[('Set A',ROOT/'outputs/v4_internal/tables/table_3_dynamic_fidelity.csv'),('Set B',ROOT/'outputs/v4_external/tables/table_3_dynamic_fidelity.csv')]):
        d=rows(path); vals=[float(next(r for r in d if r['model']==m and r['cohort']=='common_48h' and r['horizon']=='48')['change_variance_ratio_mean']) for m in MODELS]
        c.bar(x+offset,vals,width,label=scope,color=[COLOR[m] for m in MODELS],alpha=.58 if scope=='Set A' else .95,edgecolor='white')
    c.axhline(1,color='#111827',ls='--',lw=1); c.set_yscale('log'); c.set_xticks(x, [LABEL[m] for m in MODELS]); c.set_ylabel('Predicted / observed change variance (log scale)'); c.set_title('48 h temporal variability'); c.legend(frameon=False); c.grid(axis='y',alpha=.18)
    fig.suptitle('Temporal-change accuracy and variability preservation',weight='bold'); fig.tight_layout(); save(fig,'figure3_temporal_dynamics.pdf')


def coverage_panel(ax, path: Path, title: str) -> None:
    d=rows(path); nominal=np.array([.5,.8,.95])
    for m in MODELS:
        r=next(x for x in d if x['model']==m and x['cohort']=='common_48h' and x['horizon']=='48')
        obs=[float(r[f'coverage{int(n*100)}_mean']) for n in nominal]
        ax.plot(nominal,obs,marker='o',lw=1.5,color=COLOR[m],label=LABEL[m])
    ax.plot([.45,1],[.45,1],ls='--',color='#111827',lw=1); ax.set(xlim=(.47,.98),ylim=(0,.98),xlabel='Nominal coverage',ylabel='Empirical coverage',title=title); ax.grid(alpha=.18)


def figure4() -> None:
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2)); coverage_panel(axes[0],ROOT/'outputs/v4_internal/tables/table_4_reliability.csv','Locked Set A'); coverage_panel(axes[1],ROOT/'outputs/v4_external/tables/table_4_reliability.csv','Held-out Set B'); axes[1].legend(frameon=False); fig.suptitle('Conditional predictive interval coverage at 48 h',weight='bold'); fig.tight_layout(); save(fig,'figure4_interval_coverage.pdf')


def density_panel(ax,path:Path,title:str,external=False)->None:
    d=rows(path); groups=['jbi_core_dense','other_le_80pct_missing','sparse_gt80_to_95pct_missing','very_sparse_gt95pct_missing']; gl=['Core dense','Other <=80%','80-95%','>95%']
    comps=['gru_minus_linear_ar','transformer_minus_linear_ar','rssm_minus_linear_ar']; offs=[-.22,0,.22]
    key='point_estimate' if external else 'difference'
    for comp,off,m in zip(comps,offs,['gru','transformer','rssm']):
        rs=[next(r for r in d if r['density_group']==g and r['comparison']==comp) for g in groups]; y=np.arange(4)+off; val=np.array([float(r[key]) for r in rs]); lo=val-np.array([float(r['ci_low']) for r in rs]); hi=np.array([float(r['ci_high']) for r in rs])-val
        ax.errorbar(val,y,xerr=np.array([lo,hi]),fmt='o',capsize=2,color=COLOR[m],label=LABEL[m])
    ax.axvline(0,color='#111827',lw=1); ax.set_yticks(range(4),gl); ax.invert_yaxis(); ax.set(xlabel='48 h nMacroMAE difference vs Linear AR',title=title); ax.grid(axis='x',alpha=.18)


def figure5() -> None:
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True); density_panel(axes[0],ROOT/'outputs/v4_internal/tables/jbi_density_paired_ci.csv','Locked Set A'); density_panel(axes[1],ROOT/'outputs/v4_external/tables/E6_density_paired_ci.csv','Held-out Set B',True); axes[1].legend(frameon=False); fig.suptitle('Observation-density sensitivity at 48 h',weight='bold'); fig.tight_layout(); save(fig,'figure5_observation_density.pdf')


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); style(); figure1(); figure2(); figure3(); figure4(); figure5()
    print(f"generated 5 figures in {OUT}")


if __name__ == '__main__': main()
