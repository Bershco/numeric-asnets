#!/usr/bin/env python3
"""Build the pre-review advisor figures from locally cached experiment data."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "advisor_meeting_20260910" / "before_review"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime", "delivery", "tpp", "zenotravel"]
LABEL = {"block_grouping": "Block Grouping", "drone": "Drone", "fo_counters": "FO Counters", "rover": "Rover", "counters": "Counters", "mprime": "MPrime", "delivery": "Delivery", "tpp": "TPP", "zenotravel": "Zenotravel"}
TOTAL = defaultdict(lambda: 20, {"counters": 59})
BLUE, ORANGE, GREEN, RED = "#1769aa", "#e07a1f", "#238b45", "#c43c39"
PURPLE, INK, MUTED, GRID = "#7b4ab5", "#17212b", "#536273", "#d9e0e7"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def esc(v: object) -> str:
    return html.escape(str(v))


def start(w: int, h: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#536273}.label{font-size:12px}.small{font-size:10px;fill:#536273}.value{font-size:12px;font-weight:700}.big{font-size:15px;font-weight:700}</style>',
        f'<text x="34" y="39" class="title">{esc(title)}</text>',
        f'<text x="34" y="63" class="sub">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, v: object, cls: str = "label", anchor: str = "start", fill: str | None = None) -> str:
    style = f' style="fill:{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{style}>{esc(v)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", radius: float = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = "white") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}"/>'


def poly(points: list[tuple[float, float]], color: str, width: float = 2.2, dash: str = "") -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{extra}/>'


# 1. Domain scorecard: paper, best Stage-1 policy, best observed configuration.
scorecard = read(TRACK / "best_configuration_by_domain_latest.csv")
parts = start(1320, 680, "1 — best demonstrated result by domain", "Scores are normalized to test-set coverage. The best observed bar may be policy, fixed MCTS, or PW; MPrime remains provisional.")
x0, x1, y0, rh = 300, 1220, 112, 56
for tick in (0, 25, 50, 75, 100):
    xx = x0 + tick / 100 * (x1 - x0)
    parts.append(line(xx, y0 - 20, xx, y0 + rh * 9 - 10, GRID))
    parts.append(text(xx, y0 - 28, f"{tick}%", "small", "middle"))
score_rows = []
for i, row in enumerate(scorecard):
    d = row["domain"]; total = float(row["capacity"]); y = y0 + i * rh
    paper = 100 * float(row["paper_reported"]) / total
    s1 = 100 * float(row["matched_stage1_policy"]) / total
    best = 100 * float(row["result_6h_or_policy"]) / total
    parts.append(text(x0 - 15, y + 24, LABEL[d], "label", "end"))
    for offset, val, color, name in ((0, paper, "#9aa5b1", "paper"), (11, s1, ORANGE, "Stage 1"), (22, best, GREEN, "best")):
        parts.append(rect(x0, y + offset, (x1 - x0) * val / 100, 9, color, radius=2))
    parts.append(text(x1 + 10, y + 24, f"{float(row['result_6h_or_policy']):.1f}/{int(total)}  {row['best_current_configuration']}", "small"))
    score_rows.append({**row, "paper_percent": paper, "stage1_percent": s1, "best_percent": best})
parts += [
    rect(300, 632, 18, 9, "#9aa5b1"), text(325, 641, "paper", "small"),
    rect(390, 632, 18, 9, ORANGE), text(415, 641, "best Stage-1 policy", "small"),
    rect(540, 632, 18, 9, GREEN), text(565, 641, "best demonstrated result", "small"),
    text(800, 641, "Main wins: Drone, FO Counters, and Counters exceed their matched policy baselines; Zenotravel exceeds the paper mean.", "sub"), "</svg>"
]
(OUT / "01_domain_scorecard.svg").write_text("".join(parts), encoding="utf-8")
write(OUT / "01_domain_scorecard.csv", score_rows)


# 2. Full Stage-1 and Stage-2 learning dynamics for the six imperfect domains.
s1raw = read(TRACK / "learning_curves" / "stage1_policy_curve_rows_20260909.csv")
s2agg = read(TRACK / "learning_curves" / "latest" / "rq1_rq3_five_domain_learning_curve_aggregates.csv")
mps1 = read(TRACK / "mprime_validation_ipc_scale_v1" / "corrected_learning_curve_aggregate.csv")
mps2raw = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage2_checkpoints_20260903.csv")

def aggregate(rows: list[dict[str, str]], epoch_key: str, score_fn) -> list[dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in rows: grouped[int(r[epoch_key])].append(float(score_fn(r)))
    return [{"epoch": e, "mean": sum(v)/len(v), "min": min(v), "max": max(v), "n": len(v)} for e, v in sorted(grouped.items())]

curve_rows = []
parts = start(1440, 850, "2 — Stage 1 and Stage 2 learning dynamics", "Both learning curves are shown. The dotted divider is the training-stage boundary; shading is the full observed min–max range.")
for panel, domain in enumerate(["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"]):
    col, rr = panel % 3, panel // 3
    px, py, pw, ph = 55 + col * 465, 110 + rr * 345, 425, 275
    parts.append(rect(px, py, pw, ph, "#fbfcfd", GRID, 4)); parts.append(text(px+10, py+22, LABEL[domain], "big"))
    left, right, top, bottom = px+43, px+pw-12, py+38, py+ph-34; boundary=(left+right)/2
    for tick in (0,25,50,75,100):
        yy=bottom-tick/100*(bottom-top); parts.append(line(left,yy,right,yy,GRID)); parts.append(text(left-7,yy+4,tick,"small","end"))
    parts.append(line(boundary, top, boundary, bottom, INK, 1.4, "4 4"))
    parts.append(text((left+boundary)/2,bottom+20,"Stage 1 epoch", "small","middle")); parts.append(text((boundary+right)/2,bottom+20,"Stage 2 epoch", "small","middle"))
    for vh,color,rq in (("off",BLUE,"RQ1"),("on",ORANGE,"RQ3")):
        if domain == "mprime":
            one=[{"epoch":int(r["epoch"]),"mean":float(r["test_mean_pct"]),"min":float(r["test_min_pct"]),"max":float(r["test_max_pct"]),"n":int(r["n_runs"])} for r in mps1 if r["value_head"]==vh]
            two=aggregate([r for r in mps2raw if r["branch"]=="validation_led" and r["value_head"]==vh],"epoch",lambda r:100*float(r["test_success"])/20)
        else:
            raw=[r for r in s1raw if r["domain"]==domain and r["value_head"]==vh]
            one=aggregate(raw,"epoch",lambda r:100*float(r["successes"])/float(r["total"]))
            two=[{"epoch":int(r["epoch"]),"mean":float(r["mean_percent"]),"min":float(r["minimum_percent"]),"max":float(r["maximum_percent"]),"n":int(r["n_runs"])} for r in s2agg if r["domain"]==domain and r["research_question"]==rq]
        for stage,data,xa,xb in (("stage1",one,left,boundary-8),("stage2",two,boundary+8,right)):
            if not data: continue
            emin,emax=min(r["epoch"] for r in data),max(r["epoch"] for r in data)
            sx=lambda e,emin=emin,emax=emax,xa=xa,xb=xb: xa+(e-emin)/max(emax-emin,1)*(xb-xa)
            sy=lambda s: bottom-s/100*(bottom-top)
            upper=[(sx(r["epoch"]),sy(r["max"])) for r in data]; lower=[(sx(r["epoch"]),sy(r["min"])) for r in reversed(data)]
            polygon=" ".join(f"{x:.1f},{y:.1f}" for x,y in upper+lower)
            parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.09"/>'); parts.append(poly([(sx(r["epoch"]),sy(r["mean"])) for r in data],color,2.1))
            for r in data: curve_rows.append({"domain":domain,"value_head":vh,"stage":stage,**r})
parts += [line(65,815,100,815,BLUE,3),text(108,819,"VH-off mean","small"),line(230,815,265,815,ORANGE,3),text(273,819,"VH-on mean","small"),text(430,819,"Stage 1 uses ten-seed every-five policy evaluations; Stage 2 shows the evaluated refinement trajectories.","sub"),"</svg>"]
(OUT / "02_two_stage_learning_dynamics.svg").write_text("".join(parts),encoding="utf-8"); write(OUT/"02_two_stage_learning_dynamics.csv",curve_rows)


# 3. Policy-to-MCTS effects at all three per-instance cutoffs.
mcts_rows=[]
for stage,path in (("Stage 1",TRACK/"stage1_policy_mcts_all_cutoff_statistics_latest.csv"),("Stage 2",TRACK/"stage2_policy_mcts_all_cutoff_statistics_latest.csv")):
    for r in read(path): mcts_rows.append({"stage":stage,**r})
parts=start(1540,1450,"3 — what MCTS changes, and how much time it needs","Each point is the paired MCTS−policy mean; lines are 95% CIs. Stars mark Holm p<.05 within the reported family.")
left,right,top=405,1450,115
scale=lambda v:left+(v+12)/24*(right-left)
for tick in (-10,-5,0,5,10):
    xx=scale(tick);parts.append(line(xx,top,xx,1390,RED if tick==0 else GRID,1.5 if tick==0 else 1));parts.append(text(xx,94,f"{tick:+d}","small","middle"))
y=125
forest_rows=[]
for stage in ("Stage 1","Stage 2"):
    parts.append(text(35,y+7,stage,"big"));y+=28
    rows=[r for r in mcts_rows if r["stage"]==stage]
    for r in rows:
        branch=r.get("stage2_branch",r.get("experiment_id","selected")).replace("_led","").replace("MAIN-VAL","selected")
        label=f"{LABEL[r['domain']]} / VH-{r['value_head']} / {branch} / {r['search']}"
        parts.append(text(392,y+4,label,"small","end"))
        for cutoff,color,dy in (("30m",PURPLE,-6),("2h",ORANGE,0),("6h",GREEN,6)):
            change=float(r[f"change_{cutoff}"]);lo=float(r[f"ci95_low_{cutoff}"]);hi=float(r[f"ci95_high_{cutoff}"])
            p=float(r.get(f"holm_p_{cutoff}",r.get(f"holm_p_{cutoff}_interim_complete_cells",r[f"raw_p_{cutoff}"])))
            parts.append(line(scale(lo),y+dy,scale(hi),y+dy,color,2));parts.append(circle(scale(change),y+dy,4.2,color))
            if p<.05: parts.append(text(scale(change)+7,y+dy+3,"*","value",fill=color))
            forest_rows.append({"stage":stage,"branch":branch,"domain":r["domain"],"value_head":r["value_head"],"search":r["search"],"cutoff":cutoff,"policy_mean":r["policy_mean"],"mcts_mean":r[f"mcts_mean_{cutoff}"],"change":change,"ci95_low":lo,"ci95_high":hi,"raw_p":r[f"raw_p_{cutoff}"],"holm_p":p,"row_level_provenance":r["row_level_provenance"]})
        y+=35
    y+=20
parts += [line(55,1410,85,1410,PURPLE,3),text(92,1414,"30m","small"),line(150,1410,180,1410,ORANGE,3),text(187,1414,"2h","small"),line(240,1410,270,1410,GREEN,3),text(277,1414,"6h","small"),text(405,1414,"Right of zero helps; left hurts. Strongest repeatable wins are Drone/VH-on and FO Counters.","sub"),"</svg>"]
(OUT/"03_mcts_cutoff_forest.svg").write_text("".join(parts),encoding="utf-8");write(OUT/"03_mcts_cutoff_forest.csv",forest_rows)


# 4. PRESERVE-3 validation-led seed robustness only.
pres=read(TRACK/"four_domain_preservation"/"stable_domain_stage2_seed_pairs_20260902.csv")
parts=start(1280,650,"4 — PRESERVE-3 validation-led robustness","Each dot is one matched seed's Stage-2−Stage-1 change. The mean is a black tick; the TPP/off catastrophic seed is labelled explicitly.")
px,py,pw=265,125,820
for tick in (-12,-8,-4,0,4,8):
    xx=px+(tick+12)/20*pw;parts.append(line(xx,py,xx,530,RED if tick==0 else GRID,1.6 if tick==0 else 1));parts.append(text(xx,552,f"{tick:+d}","small","middle"))
for i,(d,vh) in enumerate((d,v) for d in ("delivery","tpp","zenotravel") for v in ("off","on")):
    yy=py+30+i*65;rows=[r for r in pres if r["domain"]==d and r["value_head"]==vh];vals=[float(r["change"]) for r in rows];mean=sum(vals)/len(vals)
    parts.append(text(px-15,yy+4,f"{LABEL[d]} / VH-{vh}","label","end"))
    for j,r in enumerate(rows):
        val=float(r["change"]);xx=px+(val+12)/20*pw;jitter=((j%5)-2)*3;color=RED if val<=-5 else (GREEN if val>0 else "#8793a0")
        parts.append(circle(xx,yy+jitter,5,color))
        if val<=-5: parts.append(text(xx+8,yy+jitter-8,f"seed {r['seed']}: {r['stage1_selected']}→{r['stage2_selected']}","small"))
    mx=px+(mean+12)/20*pw;parts.append(line(mx,yy-17,mx,yy+17,INK,3));parts.append(text(mx+5,yy+22,f"mean {mean:+.1f}","small"))
parts += [text(px,596,"Conclusion: Delivery and Zenotravel are preserved; TPP is preserved in 19/20 cells except one severe VH-off seed failure.","sub"),"</svg>"]
(OUT/"04_preserve3_validation_seed_robustness.svg").write_text("".join(parts),encoding="utf-8");write(OUT/"04_preserve3_validation_seed_robustness.csv",pres)


# 5. MPrime selection regret and saturation in plain terms.
s1=read(TRACK/"mprime_validation_ipc_scale_v1"/"validation_adequacy_phase_a_stage1_seeds_20260903.csv")
s2=read(TRACK/"mprime_validation_ipc_scale_v1"/"validation_adequacy_phase_a_stage2_seeds_20260903.csv")
groups=[]
for vh in ("off","on"):groups.append((f"Stage 1 / VH-{vh}",[r for r in s1 if r["value_head"]==vh]))
for branch in ("validation_led","terminal_led"):
    for vh in ("off","on"):groups.append((f"Stage 2 {branch.split('_')[0]} / VH-{vh}",[r for r in s2 if r["branch"]==branch and r["value_head"]==vh]))
summary=[]
for label,rows in groups:
    selected=sum(float(r["selected_test_score"]) for r in rows)/len(rows);best=sum(float(r["observed_test_best_score"]) for r in rows)/len(rows);frac=sum(float(r["validation_max_fraction"]) for r in rows)/len(rows)
    summary.append({"group":label,"n":len(rows),"selected_test_mean":selected,"observed_best_test_mean":best,"selection_regret":best-selected,"fraction_checkpoints_tied_at_validation_max":frac,"source":rows[0].get("source_checkpoint_audit",rows[0].get("checkpoint_ledger",""))})
parts=start(1280,620,"5 — why MPrime needs a better validation set","Orange is the test score of the checkpoint chosen by validation. Blue is the best test score observed retrospectively; the gap is selection regret, not a trainable oracle.")
px,py,pw=330,120,560
for tick in range(8,21,2):
    xx=px+(tick-8)/12*pw;parts.append(line(xx,py,xx,465,GRID));parts.append(text(xx,490,tick,"small","middle"))
for i,r in enumerate(summary):
    yy=py+30+i*53;sx=px+(r["selected_test_mean"]-8)/12*pw;bx=px+(r["observed_best_test_mean"]-8)/12*pw
    parts.append(text(px-15,yy+4,r["group"],"label","end"));parts.append(line(sx,yy,bx,yy,"#8b98a5",4));parts.append(circle(sx,yy,7,ORANGE));parts.append(circle(bx,yy,7,BLUE));parts.append(text(920,yy+4,f"loses {r['selection_regret']:.1f} plans; {r['fraction_checkpoints_tied_at_validation_max']:.0%} tied at validation max","small"))
parts += [circle(335,545,7,ORANGE),text(350,549,"validation-selected test score","small"),circle(545,545,7,BLUE),text(560,549,"best observed test score","small"),text(810,549,"Stage 2: 100% tied means validation supplied no checkpoint ranking at all.","sub"),"</svg>"]
(OUT/"05_mprime_validation_problem.svg").write_text("".join(parts),encoding="utf-8");write(OUT/"05_mprime_validation_problem.csv",summary)


(OUT/"README.md").write_text("""# Advisor figures — before independent review

This is the first complete advisor-facing set.  Every plot has a colocated CSV
with the exact local source and, where applicable, row-level job/log provenance.
The files are intentionally retained unchanged so they can be compared with the
post-review revision.
""",encoding="utf-8")
print(OUT)
