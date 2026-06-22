"""
Generate assets/results.svg — the results plot embedded in the README.
Pure stdlib (no matplotlib): emits a hand-laid SVG with two bar panels.
Re-run after adding a version:  python make_results_plot.py
"""
from pathlib import Path

# (version, test accuracy %, dataset) — classifier versions only
ACC = [("v0.2", 100.0, "4-shape"), ("v0.3", 74.6, "MNIST"),
       ("v0.5", 81.5, "MNIST"), ("v0.6", 74.4, "MNIST"),
       ("v0.8", 82.3, "tuned"), ("v0.9", 76.0, "BindsNET*"),
       ("v0.16", 68.8, "latency"), ("v0.17", 76.0, "latency+")]
# (version, compute reduction x vs dense baseline)
COMP = [("v0.3", 23.5), ("v0.5", 23.6), ("v0.4", 4.0), ("v0.6", 3.0)]

W, H = 780, 400
PLOT_TOP, PLOT_BOT = 70, 320
PLOT_H = PLOT_BOT - PLOT_TOP
ACC_COL, COMP_COL, GRID, TXT = "#2a9d8f", "#e9a23b", "#d9d9d9", "#333333"


def bars(items, x0, panel_w, vmax, color, fmt):
    n = len(items)
    slot = panel_w / n
    bw = slot * 0.56
    out = []
    for i, (ver, val, *rest) in enumerate(items):
        h = (val / vmax) * PLOT_H
        x = x0 + i * slot + (slot - bw) / 2
        y = PLOT_BOT - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" font-size="13" font-weight="bold" '
                   f'text-anchor="middle" fill="{TXT}">{fmt(val)}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{PLOT_BOT+18:.1f}" font-size="13" '
                   f'text-anchor="middle" fill="{TXT}">{ver}</text>')
        if rest:
            out.append(f'<text x="{x+bw/2:.1f}" y="{PLOT_BOT+33:.1f}" font-size="10" '
                       f'text-anchor="middle" fill="#888">{rest[0]}</text>')
    return "\n".join(out)


def panel(title, x0, panel_w, axis_label):
    return (f'<text x="{x0+panel_w/2:.1f}" y="46" font-size="15" font-weight="bold" '
            f'text-anchor="middle" fill="#222">{title}</text>'
            f'<text x="{x0+panel_w/2:.1f}" y="{H-8}" font-size="11" '
            f'text-anchor="middle" fill="#888">{axis_label}</text>'
            f'<line x1="{x0}" y1="{PLOT_BOT}" x2="{x0+panel_w}" y2="{PLOT_BOT}" stroke="{GRID}" stroke-width="1.5"/>')


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">']
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff" stroke="#e2e2e2"/>')
svg.append('<text x="20" y="26" font-size="16" font-weight="bold" fill="#111">Spiking Neural Data Lake — results by version</text>')

# Panel A: accuracy
ax0, aw = 40, 330
svg.append(panel("Test accuracy (%)", ax0, aw, "classifier versions  (MNIST chance = 10%)"))
chance_y = PLOT_BOT - (10 / 100) * PLOT_H
svg.append(f'<line x1="{ax0}" y1="{chance_y:.1f}" x2="{ax0+aw}" y2="{chance_y:.1f}" stroke="#c0392b" '
           f'stroke-width="1" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{ax0+aw-2}" y="{chance_y-4:.1f}" font-size="9" text-anchor="end" fill="#c0392b">chance 10%</text>')
svg.append(bars(ACC, ax0, aw, 100.0, ACC_COL, lambda v: f"{v:.1f}"))

# Panel B: compute reduction
bx0, bw_ = 420, 330
svg.append(panel("Compute reduction (x)", bx0, bw_, "spiking vs dense baseline  (higher = less power)"))
svg.append(bars(COMP, bx0, bw_, 26.0, COMP_COL, lambda v: f"{v:.0f}x"))

svg.append("</svg>")
out = Path("assets/results.svg")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
