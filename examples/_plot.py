"""Gaya grafik bersama: latar putih, teks hitam, aksen biru #0000ff dan merah #ff0000."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, RED, BLACK, GREY = "#0000ff", "#ff0000", "#000000", "#8a8aa8"
OUT = Path(__file__).resolve().parent.parent / "outputs"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
    "xtick.color": BLACK, "ytick.color": BLACK, "axes.grid": True,
    "grid.color": "#e3e3ef", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, RED, BLACK, "#7f7fff", "#ff8080"]),
})


def fig(w=6.4, h=3.4, **kw):
    return plt.subplots(figsize=(w, h), **kw)


def save(name):
    OUT.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=110)
    plt.close("all")
