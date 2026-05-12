"""Publication-quality architecture diagrams using matplotlib.

Style references:
- U-Net (Ronneberger 2015): sized blocks forming U-shape
- SwinIR (Liang 2021): horizontal pipeline with zoom panels

Run: python scripts/gen_arch_diagrams.py

Generates:
  figures/arch_flow_unet_eval.png
  figures/arch_swinir_eval.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

DPI = 300

# Muted, colorblind-friendly palette
P = dict(
    feat="#7BAED4", feat_e="#4A80A8",
    bot="#E89060",  bot_e="#C07040",
    time="#B09BD0", time_e="#8070A8",
    conv="#7BAED4", conv_e="#4A80A8",
    norm="#B09BD0", norm_e="#8070A8",
    samp="#6AAF7A", samp_e="#4A8A5A",
    io="#E0E0E0",   io_e="#BBBBBB",
    add="#D4B050",  add_e="#B09030",
    down="#B05050",
    up="#50A068",
    skip="#AAAAAA",
    flow="#555555",
    text="#333333",  dim="#888888",
    zoom="#CCCCCC",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


# ── Drawing primitives ──

def _blk(ax, cx, cy, w, h, label, sub="", *,
         fc, ec, tc="white", fs=8, sfs=6, lw=0.7):
    """Rounded-rect block with centered label and optional subtitle."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015", fc=fc, ec=ec, lw=lw, zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.15, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - h * 0.18, sub, ha="center", va="center",
                fontsize=sfs, color=tc, alpha=0.9, zorder=3)
    elif label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)


def _arr(ax, p0, p1, c="#555", lw=1.0, ls="-", sty="-|>",
         ms=10, cs="arc3,rad=0"):
    """Arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=sty, mutation_scale=ms,
        lw=lw, color=c, connectionstyle=cs, linestyle=ls, zorder=1))


def _circ(ax, cx, cy, r, label="+"):
    """Small circle node for residual addition."""
    ax.add_patch(Circle(
        (cx, cy), r, fc=P["add"], ec=P["add_e"], lw=0.7, zorder=2))
    ax.text(cx, cy, label, fontsize=max(6, int(r * 45)),
            ha="center", va="center", fontweight="bold", color="#333",
            zorder=3)


def _chain(ax, nodes, y, c="#555", lw=0.7, ms=7):
    """Draw flow arrows between consecutive (x_center, half_extent) nodes."""
    gap = 0.03
    for i in range(len(nodes) - 1):
        x0, r0 = nodes[i]
        x1, r1 = nodes[i + 1]
        _arr(ax, (x0 + r0 + gap, y), (x1 - r1 - gap, y),
             c=c, lw=lw, ms=ms)


# ═══════════════════════════════════════════════════════════════════
#  AttentionUNet — Flow Matching Velocity Predictor
# ═══════════════════════════════════════════════════════════════════

def gen_flow_unet(path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Block width ∝ channels; U-shape from encoder/decoder x-shifts
    CH_W = {96: 1.0, 192: 1.5, 384: 2.0}
    BH = 0.8
    Y = [7.5, 5.5, 3.8]
    XE = [2.0, 2.5, 3.0]
    XD = [8.0, 7.5, 7.0]
    CH = [96, 192, 384]
    RES = ["128²", "64²", "32²"]
    XB, YB, WB, HB = 5.0, 2.2, 1.8, 0.65

    # Title
    ax.text(5, 9.25, "AttentionUNet", fontsize=13, fontweight="bold",
            ha="center", color=P["text"])
    ax.text(5, 8.9,
            "Flow Matching Velocity Predictor  ·  base=96  ·  "
            "mults=(1,2,4)  ·  4-head attn @ 16²",
            fontsize=6.5, ha="center", color=P["dim"])

    # ── Encoder blocks ──
    for i in range(3):
        _blk(ax, XE[i], Y[i], CH_W[CH[i]], BH,
             "2× ResBlock", f"{CH[i]} ch, {RES[i]}",
             fc=P["feat"], ec=P["feat_e"], fs=7.5, sfs=6)

    # ── Decoder blocks ──
    for i in range(3):
        _blk(ax, XD[i], Y[i], CH_W[CH[i]], BH,
             "2× ResBlock", f"{CH[i]} ch, {RES[i]}",
             fc=P["feat"], ec=P["feat_e"], fs=7.5, sfs=6)

    # ── Bottleneck ──
    _blk(ax, XB, YB, WB, HB,
         "Res → Attn(4h) → Res", "384 ch, 16²",
         fc=P["bot"], ec=P["bot_e"], fs=7, sfs=5.5)

    # ── Encoder downward arrows ──
    for i in range(2):
        y0 = Y[i] - BH / 2 - 0.03
        y1 = Y[i + 1] + BH / 2 + 0.03
        _arr(ax, (XE[i], y0), (XE[i + 1], y1), c=P["down"])
        mx = (XE[i] + XE[i + 1]) / 2 + 0.55
        my = (y0 + y1) / 2
        ax.text(mx, my, "↓ Conv s2", fontsize=5.5,
                color=P["down"], va="center")

    # enc2 → bottleneck
    y0 = Y[2] - BH / 2 - 0.03
    y1 = YB + HB / 2 + 0.03
    _arr(ax, (XE[2], y0), (XB - WB / 4, y1), c=P["down"])
    ax.text((XE[2] + XB) / 2 - 0.1, (y0 + y1) / 2 + 0.15,
            "↓ Conv s2", fontsize=5.5, color=P["down"], ha="center")

    # ── Decoder upward arrows ──
    _arr(ax, (XB + WB / 4, y1), (XD[2], y0), c=P["up"])
    ax.text((XB + XD[2]) / 2 + 0.1, (y0 + y1) / 2 + 0.15,
            "↑ interp + Conv", fontsize=5.5, color=P["up"], ha="center")

    for i in range(2):
        ybot = Y[i + 1] + BH / 2 + 0.03
        ytop = Y[i] - BH / 2 - 0.03
        _arr(ax, (XD[i + 1], ybot), (XD[i], ytop), c=P["up"])
        mx = (XD[i + 1] + XD[i]) / 2 - 0.55
        my = (ybot + ytop) / 2
        ax.text(mx, my, "↑ interp + Conv", fontsize=5.5,
                color=P["up"], ha="right", va="center")

    # ── Skip connections ──
    for i in range(3):
        w = CH_W[CH[i]]
        x0 = XE[i] + w / 2 + 0.03
        x1 = XD[i] - w / 2 - 0.03
        _arr(ax, (x0, Y[i]), (x1, Y[i]),
             c=P["skip"], ls="--", lw=0.8, ms=7)
        ax.text((x0 + x1) / 2, Y[i] + 0.15, "skip + cat",
                fontsize=5, color=P["skip"], ha="center", style="italic")

    # ── Input ──
    iy = Y[0] + BH / 2
    ax.text(XE[0], iy + 0.55, "Input", fontsize=9, fontweight="bold",
            ha="center", color=P["text"])
    ax.text(XE[0], iy + 0.3,
            "LR (1×32²) → bilinear ↑4× → cat(xₜ) → Conv 3×3",
            fontsize=5, ha="center", color=P["dim"])
    _arr(ax, (XE[0], iy + 0.18), (XE[0], iy + 0.03),
         c=P["flow"], lw=0.7, ms=7)

    # ── Output ──
    ax.text(XD[0], iy + 0.55, "Output", fontsize=9, fontweight="bold",
            ha="center", color=P["text"])
    ax.text(XD[0], iy + 0.3,
            "GN → SiLU → Conv 1×1 → v(xₜ, t)",
            fontsize=5, ha="center", color=P["dim"])
    _arr(ax, (XD[0], iy + 0.03), (XD[0], iy + 0.18),
         c=P["flow"], lw=0.7, ms=7)

    # ── Time conditioning (bus to ALL ResBlocks + bottleneck) ──
    tx, ty = 5.0, 8.4
    _blk(ax, tx, ty, 2.6, 0.4,
         "t → SinEmb → MLP → 256d", "",
         fc=P["time"], ec=P["time_e"], fs=6.5)
    ax.text(tx, ty + 0.3, "Time Conditioning", fontsize=7,
            fontweight="bold", ha="center", color=P["time_e"])

    tc = P["time_e"]
    tlw = 0.45

    # Left bus — runs outside encoder column, stubs into each block
    bus_lx = 1.0
    ax.plot([tx - 1.0, bus_lx], [ty - 0.22, iy],
            ls=":", lw=tlw, c=tc, zorder=0)
    ax.plot([bus_lx, bus_lx], [iy, Y[2] - BH / 2],
            ls=":", lw=tlw, c=tc, zorder=0)
    ax.plot([bus_lx, XB - WB / 2 - 0.03], [Y[2] - BH / 2, YB + HB / 2],
            ls=":", lw=tlw, c=tc, zorder=0)
    for i in range(3):
        w = CH_W[CH[i]]
        _arr(ax, (bus_lx, Y[i]), (XE[i] - w / 2 - 0.03, Y[i]),
             c=tc, ls=":", lw=tlw, sty="->", ms=3)

    # Right bus — runs outside decoder column
    bus_rx = 9.0
    ax.plot([tx + 1.0, bus_rx], [ty - 0.22, iy],
            ls=":", lw=tlw, c=tc, zorder=0)
    ax.plot([bus_rx, bus_rx], [iy, Y[2] - BH / 2],
            ls=":", lw=tlw, c=tc, zorder=0)
    ax.plot([bus_rx, XB + WB / 2 + 0.03], [Y[2] - BH / 2, YB + HB / 2],
            ls=":", lw=tlw, c=tc, zorder=0)
    for i in range(3):
        w = CH_W[CH[i]]
        _arr(ax, (bus_rx, Y[i]), (XD[i] + w / 2 + 0.03, Y[i]),
             c=tc, ls=":", lw=tlw, sty="->", ms=3)

    ax.text(bus_lx - 0.08, (Y[0] + Y[2]) / 2, "t_emb (scale+shift)",
            fontsize=4.5, color=tc, ha="right", va="center",
            rotation=90, style="italic")

    # ── Legend ──
    ly = 0.6
    for j, (color, label) in enumerate([
        (P["feat"], "2× ResBlock"),
        (P["bot"], "Bottleneck (Self-Attn)"),
        (P["time"], "Time conditioning"),
    ]):
        lx = 1.8 + j * 2.5
        ax.add_patch(Rectangle((lx, ly - 0.08), 0.2, 0.16,
                               fc=color, ec="none", zorder=2))
        ax.text(lx + 0.3, ly, label, fontsize=6, va="center",
                color=P["text"])

    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  SwinIR-M — single axes, three panels with zoom indicators
# ═══════════════════════════════════════════════════════════════════

def gen_swinir(path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    bw1, a_bh, r1 = 0.82, 0.65, 0.18
    bw2, b_bh, r2 = 0.75, 0.50, 0.15
    bw3, c_bh, r3 = 0.75, 0.50, 0.14

    # ═══ (a) Main pipeline  ─────────────────────────────────────
    a_by = 7.2
    ax.text(0.2, 8.7,
            "(a)  SwinIR-M — Finetuned 1ch Climate SR ×4",
            fontsize=10, fontweight="bold", color=P["text"])
    ax.text(0.2, 8.3,
            "embed=96  ·  depths=[6,6,6,6]  ·  heads=[6,6,6,6]  "
            "·  window=8  ·  3ch → 1ch adapted",
            fontsize=6, color=P["dim"])

    main = [
        (0.6,  0.30, "LR",           "1×H×W",   P["io"],   P["io_e"]),
        (1.7,  bw1/2,"Conv",         "1 → 96",  P["conv"], P["conv_e"]),
        (3.0,  bw1/2,"RSTB₁",        "6×STL",   P["bot"],  P["bot_e"]),
        (4.1,  bw1/2,"RSTB₂",        "6×STL",   P["bot"],  P["bot_e"]),
        (5.2,  bw1/2,"RSTB₃",        "6×STL",   P["bot"],  P["bot_e"]),
        (6.3,  bw1/2,"RSTB₄",        "6×STL",   P["bot"],  P["bot_e"]),
        (7.6,  bw1/2,"Conv",         "96 → 96", P["conv"], P["conv_e"]),
        # 8.5: + node
        (9.4,  bw1/2,"Conv",         "96 → 64", P["conv"], P["conv_e"]),
        (10.7, 0.55, "PixelShuffle", "2×(↑2)",  P["samp"], P["samp_e"]),
        (11.9, bw1/2,"Conv",         "64 → 1",  P["conv"], P["conv_e"]),
        (12.9, 0.30, "HR",           "1×4H×4W", P["io"],   P["io_e"]),
    ]
    for x, hw, lab, sub, fc, ec in main:
        _blk(ax, x, a_by, hw * 2, a_bh, lab, sub,
             fc=fc, ec=ec, fs=6.5, sfs=5)
    _circ(ax, 8.5, a_by, r1)

    chain_a = [(n[0], n[1]) for n in main[:7]]
    chain_a.append((8.5, r1))
    chain_a += [(n[0], n[1]) for n in main[7:]]
    _chain(ax, chain_a, a_by, c=P["flow"], lw=0.7, ms=7)

    # Global residual
    _arr(ax, (1.7, a_by + a_bh / 2 + 0.05),
         (8.5, a_by + a_bh / 2 + 0.05),
         c=P["skip"], ls="--", lw=0.7, ms=6, cs="arc3,rad=-0.12")
    ax.text(5.1, a_by + a_bh / 2 + 0.32,
            "global residual", fontsize=5,
            color=P["skip"], ha="center", style="italic")

    # ═══ (b) RSTB detail  ──────────────────────────────────────
    b_by = 4.0
    ax.text(0.2, 5.3,
            "(b)  RSTB — Residual Swin Transformer Block",
            fontsize=9, fontweight="bold", color=P["text"])

    rstb = [
        (0.5,  0.22, "In",   "",       P["io"],   P["io_e"]),
        (1.5,  bw2/2,"STL",  "W-MSA",  P["feat"], P["feat_e"]),
        (2.7,  bw2/2,"STL",  "SW-MSA", P["feat"], P["feat_e"]),
        (3.9,  bw2/2,"STL",  "W-MSA",  P["feat"], P["feat_e"]),
        (5.1,  bw2/2,"STL",  "SW-MSA", P["feat"], P["feat_e"]),
        (6.3,  bw2/2,"STL",  "W-MSA",  P["feat"], P["feat_e"]),
        (7.5,  bw2/2,"STL",  "SW-MSA", P["feat"], P["feat_e"]),
        (8.8,  bw2/2,"Conv", "3×3",     P["conv"], P["conv_e"]),
        (10.5, 0.22, "Out",  "",       P["io"],   P["io_e"]),
    ]
    for x, hw, lab, sub, fc, ec in rstb:
        _blk(ax, x, b_by, hw * 2, b_bh, lab, sub,
             fc=fc, ec=ec, fs=6, sfs=4.5)
    _circ(ax, 9.6, b_by, r2)

    chain_b = [(n[0], n[1]) for n in rstb[:8]]
    chain_b.append((9.6, r2))
    chain_b.append((rstb[8][0], rstb[8][1]))
    _chain(ax, chain_b, b_by, c=P["flow"], lw=0.6, ms=6)

    _arr(ax, (0.5, b_by + b_bh / 2 + 0.05),
         (9.6, b_by + b_bh / 2 + 0.05),
         c=P["skip"], ls="--", lw=0.6, ms=5, cs="arc3,rad=-0.10")
    ax.text(5.0, b_by + b_bh / 2 + 0.30,
            "residual", fontsize=5,
            color=P["skip"], ha="center", style="italic")

    # ═══ (c) STL detail  ───────────────────────────────────────
    c_by = 1.0
    ax.text(0.2, 2.3,
            "(c)  STL — Swin Transformer Layer",
            fontsize=9, fontweight="bold", color=P["text"])

    stl = [
        (0.5, 0.22,  "In",       "",           P["io"],   P["io_e"]),
        (1.6, bw3/2, "LN",       "",           P["norm"], P["norm_e"]),
        (3.0, 0.55,  "W/SW-MSA", "6h, w=8",   P["bot"],  P["bot_e"]),
        # 4.3: + node
        (5.3, bw3/2, "LN",       "",           P["norm"], P["norm_e"]),
        (6.7, bw3/2, "MLP",      "96→384→96", P["conv"], P["conv_e"]),
        # 7.8: + node
        (8.8, 0.22,  "Out",      "",           P["io"],   P["io_e"]),
    ]
    for x, hw, lab, sub, fc, ec in stl:
        _blk(ax, x, c_by, hw * 2, c_bh, lab, sub,
             fc=fc, ec=ec, fs=6, sfs=4.5)
    _circ(ax, 4.3, c_by, r3)
    _circ(ax, 7.9, c_by, r3)

    chain_c = [
        (0.5, 0.22), (1.6, bw3 / 2), (3.0, 0.55),
        (4.3, r3), (5.3, bw3 / 2), (6.7, bw3 / 2),
        (7.9, r3), (8.8, 0.22),
    ]
    _chain(ax, chain_c, c_by, c=P["flow"], lw=0.6, ms=6)

    # Residual: In → first +
    _arr(ax, (0.5, c_by + c_bh / 2 + 0.05),
         (4.3, c_by + c_bh / 2 + 0.05),
         c=P["skip"], ls="--", lw=0.6, ms=5, cs="arc3,rad=-0.08")
    ax.text(2.4, c_by + c_bh / 2 + 0.25,
            "residual", fontsize=4.5,
            color=P["skip"], ha="center", style="italic")

    # Residual: first + → second +
    _arr(ax, (4.3, c_by + c_bh / 2 + 0.05),
         (7.9, c_by + c_bh / 2 + 0.05),
         c=P["skip"], ls="--", lw=0.6, ms=5, cs="arc3,rad=-0.08")
    ax.text(6.1, c_by + c_bh / 2 + 0.25,
            "residual", fontsize=4.5,
            color=P["skip"], ha="center", style="italic")

    # ═══ Zoom indicators  ──────────────────────────────────────
    zc = "#B8B8B8"

    # RSTB₁ in (a) → panel (b)
    rstb1_bl = (3.0 - bw1 / 2, a_by - a_bh / 2)
    rstb1_br = (3.0 + bw1 / 2, a_by - a_bh / 2)
    b_tl = (0.3, b_by + b_bh / 2 + 0.45)
    b_tr = (10.7, b_by + b_bh / 2 + 0.45)
    ax.plot([rstb1_bl[0], b_tl[0]], [rstb1_bl[1], b_tl[1]],
            ls="--", lw=0.6, c=zc, zorder=0)
    ax.plot([rstb1_br[0], b_tr[0]], [rstb1_br[1], b_tr[1]],
            ls="--", lw=0.6, c=zc, zorder=0)

    # STL₁ in (b) → panel (c)
    stl1_bl = (1.5 - bw2 / 2, b_by - b_bh / 2)
    stl1_br = (1.5 + bw2 / 2, b_by - b_bh / 2)
    c_tl = (0.3, c_by + c_bh / 2 + 0.45)
    c_tr = (9.0, c_by + c_bh / 2 + 0.45)
    ax.plot([stl1_bl[0], c_tl[0]], [stl1_bl[1], c_tl[1]],
            ls="--", lw=0.6, c=zc, zorder=0)
    ax.plot([stl1_br[0], c_tr[0]], [stl1_br[1], c_tr[1]],
            ls="--", lw=0.6, c=zc, zorder=0)

    # Thin separators between panels
    ax.axhline(y=6.1, xmin=0.01, xmax=0.99,
               color="#E8E8E8", lw=0.5, zorder=0)
    ax.axhline(y=3.0, xmin=0.01, xmax=0.99,
               color="#E8E8E8", lw=0.5, zorder=0)

    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    gen_flow_unet(str(FIGURES / "arch_flow_unet_eval.png"))
    gen_swinir(str(FIGURES / "arch_swinir_eval.png"))
