import json
import pathlib

import matplotlib
matplotlib.use("Agg")            # no display needed; write straight to PNG
import matplotlib.pyplot as plt
import numpy as np

from _common import RESULTS, banner, parse_args

args = parse_args(__doc__)

# ----------------------------------------------------------------------
# Published full-mode values, used as fallbacks.
# ----------------------------------------------------------------------
FALLBACK = {
    "mechanisms": [28.5, 2.8, 1.9, -1.0],
    "asymptotic_N": [10, 20, 40, 80, 160, 320],
    "asymptotic": {
        "myopic greedy": [26.20, 51.24, 58.38, 45.70, 38.58, 37.04],
        "index": [23.56, 42.69, 52.45, 46.21, 38.57, 38.23],
        "index + travel charge": [12.58, 17.49, 19.81, 18.45, 17.99, 18.56],
    },
    "beta": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
    "blind": [34.76, 9.07, 7.24, 6.27, 18.80, 28.62],
    "aware": [45.13, 31.85, 13.05, 6.12, 3.50, 12.41],
    "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "dose": [2.25, 2.30, 2.70, 4.39, 5.66],
}


def load(name):
    """Return the parsed results file, or None if this experiment was not run."""
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def tag(live):
    """Suffix for a panel title, recording where its numbers came from."""
    return "" if live else "  [published values]"


banner("building the summary figure")

fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))

# ======================================================================
# Panel 1 -- ranking of mechanisms
# ======================================================================
d02, d03, d07 = load("e02_policy_ladder"), load("e03_diagnostics"), load("e07_staleness")
live1 = all(x is not None for x in (d02, d03, d07))
if live1:
    # travel charge, under multi-epoch travel: beta=0 minus the best beta
    b = d07["results"]["B_tuned_beta"]["rows"]
    travel = b[0]["blind"] - min(r["blind"] for r in b)
    # projection, at each variant's own best beta
    stale = float(np.mean(d07["results"]["B_tuned_beta"]["paired_diff"]))
    # index vs greedy, from regime A
    idx = float(np.mean(d02["results"]["A_separating"]["d_greedy_to_index"]))
    # matching-awareness (negated: we plot "improvement", and it is negative)
    match = float(np.mean(d02["results"]["A_separating"]
                          ["d_index_naive_to_matched"]))
    vals = [travel, stale, idx, match]
else:
    vals = FALLBACK["mechanisms"]

names = ["travel charge\nin objective", "staleness\ncorrection",
         "index vs\nmyopic greedy", "matching-aware\nindex"]
# Colour by magnitude and sign so the ordering reads at a glance.
cols = ["#1a73e8" if v > 5 else "#5f9ea0" if v > 0 else "#d93025" for v in vals]
bars = ax[0].bar(names, vals, color=cols, edgecolor="black", linewidth=0.6)
ax[0].axhline(0, color="black", lw=1)
ax[0].set_ylabel("improvement (percentage points)")
ax[0].set_title("What actually helps\n(measured against exact optima)" + tag(live1),
                fontsize=11)
for r, v in zip(bars, vals):
    ax[0].text(r.get_x() + r.get_width() / 2, v + (1.2 if v > 0 else -2.6),
               f"{v:+.1f}", ha="center", fontsize=10, fontweight="bold")
ax[0].set_ylim(min(-5, min(vals) - 3), max(33, max(vals) * 1.15))

# ======================================================================
# Panel 2 -- asymptotic behaviour
# ======================================================================
d06 = load("e06_asymptotic")
live2 = d06 is not None
if live2:
    Ns = np.array([r["N"] for r in d06["rows"]], float)
    series = {
        "myopic greedy": [r["greedy"] for r in d06["rows"]],
        "index": [r["index"] for r in d06["rows"]],
        "index + travel charge": [r["index_travel"] for r in d06["rows"]],
    }
else:
    Ns = np.array(FALLBACK["asymptotic_N"], float)
    series = FALLBACK["asymptotic"]

for k, v in series.items():
    ax[1].plot(Ns, v, "o-", lw=2, markersize=6, label=k)
ax[1].set_xscale("log", base=2)          # scaling is geometric in h
ax[1].set_xlabel("zones N   (fleet K = 0.4N, degree fixed)")
ax[1].set_ylabel("deviation from LP lower bound (%)")
# The "plateaus" claim needs enough scale points to be visible. In --quick mode
# there are only three, and the curve has not yet turned over, so asserting a
# plateau there would be dishonest.
if len(Ns) >= 5:
    _t2 = ("Deviation plateaus, it does not vanish\n"
           "(theory predicts exponential decay)")
else:
    _t2 = (f"Deviation over available scales (N<={int(Ns[-1])})\n"
           "run in full mode to see the plateau")
ax[1].set_title(_t2 + tag(live2), fontsize=11)
ax[1].legend(fontsize=8, frameon=False)
ax[1].grid(alpha=0.3)
ax[1].set_ylim(0, max(65, max(max(v) for v in series.values()) * 1.15))
# Shade what a vanishing gap would look like, so "plateau" is visually obvious.
ax[1].axhspan(0, 3, color="green", alpha=0.08)
ax[1].text(Ns[1], 1.0, "what an exponentially\nvanishing gap would look like",
           fontsize=7, color="green")

# ======================================================================
# Panel 3 -- travel charge under multi-epoch travel
# ======================================================================
live3 = d07 is not None
if live3:
    rows = d07["results"]["B_tuned_beta"]["rows"]
    beta = [r["beta"] for r in rows]
    blind = [r["blind"] for r in rows]
    aware = [r["aware"] for r in rows]
    bb, ba = (d07["results"]["B_tuned_beta"]["best_blind_beta"],
              d07["results"]["B_tuned_beta"]["best_aware_beta"])
else:
    beta, blind, aware = FALLBACK["beta"], FALLBACK["blind"], FALLBACK["aware"]
    bb, ba = 3.0, 4.0

ax[2].plot(beta, blind, "o-", lw=2, color="#1a73e8",
           label="index at current state")
ax[2].plot(beta, aware, "s-", lw=2, color="#e8710a",
           label="index projected to arrival")
# Ring the optimum of each curve: they are at DIFFERENT beta, which is the point.
ax[2].scatter([bb], [blind[beta.index(bb)]], s=160, facecolors="none",
              edgecolors="#1a73e8", lw=2)
ax[2].scatter([ba], [aware[beta.index(ba)]], s=160, facecolors="none",
              edgecolors="#e8710a", lw=2)
ax[2].set_xlabel("travel-charge weight  beta")
ax[2].set_ylabel("gap from exact optimum (%)")
ax[2].set_title("Multi-epoch travel: the charge dominates,\n"
                "and the two corrections interact" + tag(live3), fontsize=11)
ax[2].legend(fontsize=8, frameon=False)
ax[2].grid(alpha=0.3)
ax[2].annotate("over-charge ->\nvehicles freeze",
               xy=(beta[-1], aware[-1]), xytext=(beta[-2] * 0.75, max(aware) * 0.9),
               arrowprops=dict(arrowstyle="->", color="gray"),
               fontsize=8, color="gray")

# ======================================================================
# Panel 4 -- the proposed contribution, dose-response
# ======================================================================
live4 = d03 is not None
if live4:
    rows = d03["results"]["I_dose_response"]["rows"]
    alpha = [r["alpha"] for r in rows]
    dose = [r["gap"] for r in rows]
else:
    alpha, dose = FALLBACK["alpha"], FALLBACK["dose"]

ax[3].plot(alpha, dose, "o-", color="#d93025", lw=2, markersize=7)
ax[3].set_xlabel("weight on realised service rate\n(0 = classical relaxation)")
ax[3].set_ylabel("gap from exact optimum (%)")
ax[3].set_title("Making the index 'matching-aware'\nmonotonically hurts" + tag(live4),
                fontsize=11)
ax[3].grid(alpha=0.3)
ax[3].annotate("worse", xy=(alpha[-1] * 0.95, dose[-1] * 0.96),
               xytext=(0.55, dose[0] + 0.7 * (dose[-1] - dose[0])),
               arrowprops=dict(arrowstyle="->", color="#d93025"),
               color="#d93025", fontsize=10)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
RESULTS.mkdir(exist_ok=True)
out = RESULTS / "spike_findings.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[saved] {out.relative_to(RESULTS.parent)}")
if not all((live1, live2, live3, live4)):
    print("NOTE: some panels used published fallback values because the")
    print("corresponding experiment has not been run. Panel titles say which.")
