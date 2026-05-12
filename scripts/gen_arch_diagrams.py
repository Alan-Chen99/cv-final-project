"""Architecture diagrams using Graphviz for professional, aligned layout.

Replaces matplotlib FancyBboxPatch approach with Graphviz automatic layout
to eliminate overlap and alignment issues.

Generates:
  figures/arch_flow_unet_eval.png   (AttentionUNet — Flow Matching)
  figures/arch_swinir_eval.png      (SwinIR-M — Finetuned 1ch)
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

DPI = 200

# Palette
RES = "#5B9BD5"
ATTN = "#E07050"
CONV = "#3A7CC5"
SAMP = "#6AAF6E"
IO = "#D9D9D9"
SKIP = "#D4A520"
SKIP_T = "#B08A20"
BOT_BG = "#FFF5F0"
NORM = "#9B6BB5"
TIME = "#9B6BB5"


def _render(src: str, out: Path):
    out.with_suffix(".dot").write_text(src)
    subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={DPI}", str(out.with_suffix(".dot")), "-o", str(out)],
        check=True,
    )


def _render_to_image(src: str) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False, mode="w") as f:
        f.write(src)
        dot_path = f.name
    png_path = dot_path.replace(".dot", ".png")
    subprocess.run(["dot", "-Tpng", f"-Gdpi={DPI}", dot_path, "-o", png_path], check=True)
    return Image.open(png_path).convert("RGB")


def _vstack(images: list[Image.Image], gap: int = 20) -> Image.Image:
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in images:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    return out


# ══════════════════════════════════════════════════════════════════════════
#  AttentionUNet — U-shape via rank constraints + dir=back
# ══════════════════════════════════════════════════════════════════════════

def gen_flow_unet(path: str):
    src = f'''digraph AttentionUNet {{
    graph [
        rankdir=TB
        fontname="Helvetica"
        bgcolor=white
        pad="0.5"
        nodesep=1.5
        ranksep=0.5
        label=<<TABLE BORDER="0" CELLPADDING="4">
            <TR><TD><B><FONT POINT-SIZE="28">AttentionUNet</FONT></B>
            <FONT POINT-SIZE="15" COLOR="#666"> — Flow Matching Velocity Predictor</FONT></TD></TR>
            <TR><TD><FONT POINT-SIZE="13" COLOR="#777">base=96 | mults=(1,2,4) | 4-head self-attention @ 16x16 bottleneck</FONT></TD></TR>
        </TABLE>>
        labelloc=t
    ]

    node [fontname="Helvetica" fontsize=16 shape=box style="filled,rounded"
          penwidth=1.3 margin="0.22,0.1"]
    edge [fontname="Helvetica" fontsize=13 penwidth=1.5 arrowsize=0.9]

    // ═══ INPUT / OUTPUT ═══
    input [label=<<TABLE BORDER="0" CELLSPACING="1">
        <TR><TD><B>Input</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="13">LR (1x32x32) → <B>Bilinear ↑4x</B> → 1x128x128</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="13">cat(LR↑, x_t) → Conv 3x3</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="12" COLOR="#555">2 → 96 ch, 128x128</FONT></TD></TR>
    </TABLE>> fillcolor="{IO}"]

    output [label=<<TABLE BORDER="0" CELLSPACING="1">
        <TR><TD><B>Output</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="13">GroupNorm, SiLU, Conv 1x1</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="12" COLOR="#555">96 → 1 ch, velocity v(x_t, t)</FONT></TD></TR>
    </TABLE>> fillcolor="{IO}"]

    // ═══ TIME CONDITIONING ═══
    time [label=<<TABLE BORDER="0" CELLSPACING="1">
        <TR><TD><B>Time Conditioning</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="13">t ∈ [0,1] → Sinusoidal Emb</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="13">→ MLP → 256-d</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="12"><I>Scale + shift every ResBlock</I></FONT></TD></TR>
    </TABLE>> fillcolor="{TIME}" fontcolor=white shape=note]

    // ═══ ENCODER ═══
    enc0 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">96 ch, 128x128</FONT>>
          fillcolor="{RES}" fontcolor=white]
    d0 [label="Downsample: Conv stride 2"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]
    enc1 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">192 ch, 64x64</FONT>>
          fillcolor="{RES}" fontcolor=white]
    d1 [label="Downsample: Conv stride 2"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]
    enc2 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">384 ch, 32x32</FONT>>
          fillcolor="{RES}" fontcolor=white]
    d2 [label="Downsample: Conv stride 2"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]

    // ═══ BOTTLENECK (single node for centered U-bottom) ═══
    bot [label=<<TABLE BORDER="0" CELLSPACING="1">
        <TR><TD><B><FONT POINT-SIZE="17">Bottleneck</FONT></B></TD></TR>
        <TR><TD>ResBlock → <FONT COLOR="#FFD0C0"><B>Self-Attn (4 heads)</B></FONT> → ResBlock</TD></TR>
        <TR><TD><FONT POINT-SIZE="12" COLOR="#FFE0D0">384 ch, 16x16</FONT></TD></TR>
    </TABLE>> fillcolor="{ATTN}" fontcolor=white penwidth=2]

    // ═══ DECODER ═══
    u2 [label="Upsample: interp + Conv"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]
    dec2 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">384 ch, 32x32</FONT>>
          fillcolor="{RES}" fontcolor=white]
    u1 [label="Upsample: interp + Conv"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]
    dec1 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">192 ch, 64x64</FONT>>
          fillcolor="{RES}" fontcolor=white]
    u0 [label="Upsample: interp + Conv"
        fillcolor="{SAMP}" fontcolor=white fontsize=14 margin="0.12,0.04"]
    dec0 [label=<<B>2x ResBlock</B><BR/><FONT POINT-SIZE="13">96 ch, 128x128</FONT>>
          fillcolor="{RES}" fontcolor=white]

    // ═══ ENCODER FLOW (downward) ═══
    input -> enc0 [weight=8]
    enc0 -> d0 [weight=8]
    d0 -> enc1 [weight=8]
    enc1 -> d1 [weight=8]
    d1 -> enc2 [weight=8]
    enc2 -> d2 [weight=8]
    d2 -> bot [weight=8]

    // ═══ DECODER FLOW (upward via dir=back) ═══
    output -> dec0 [dir=back weight=8]
    dec0 -> u0 [dir=back weight=8]
    u0 -> dec1 [dir=back weight=8]
    dec1 -> u1 [dir=back weight=8]
    u1 -> dec2 [dir=back weight=8]
    dec2 -> u2 [dir=back weight=8]
    u2 -> bot [dir=back weight=8]

    // ═══ RANK CONSTRAINTS (U-shape) ═══
    {{rank=same; input; output; time}}
    {{rank=same; enc0; dec0}}
    {{rank=same; d0; u0}}
    {{rank=same; enc1; dec1}}
    {{rank=same; d1; u1}}
    {{rank=same; enc2; dec2}}
    {{rank=same; d2; u2}}

    // ═══ TIME LAYOUT ═══
    output -> time [style=invis]
    time -> enc1 [style=dotted color="{TIME}" constraint=false penwidth=1.5
        label=<<FONT POINT-SIZE="11" COLOR="{TIME}"><I>to all ResBlocks</I></FONT>>]

    // ═══ SKIP CONNECTIONS ═══
    enc0 -> dec0 [constraint=false style=dashed color="{SKIP}" penwidth=2.5
        label=<<FONT COLOR="{SKIP_T}"><B>skip + concat</B></FONT>>]
    enc1 -> dec1 [constraint=false style=dashed color="{SKIP}" penwidth=2.5
        label=<<FONT COLOR="{SKIP_T}"><B>skip + concat</B></FONT>>]
    enc2 -> dec2 [constraint=false style=dashed color="{SKIP}" penwidth=2.5
        label=<<FONT COLOR="{SKIP_T}"><B>skip + concat</B></FONT>>]
}}'''

    out = Path(path)
    _render(src, out)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
#  SwinIR-M Finetuned — 3 stacked horizontal diagrams
# ══════════════════════════════════════════════════════════════════════════

def gen_swinir(path: str):
    # Part 1: Main pipeline (left to right)
    main = f'''digraph SwinIR_Main {{
    graph [rankdir=LR fontname="Helvetica" bgcolor=white pad="0.4"
           nodesep=0.35 ranksep=0.45
        label=<<TABLE BORDER="0" CELLPADDING="4">
            <TR><TD><B><FONT POINT-SIZE="26">SwinIR-M</FONT></B>
            <FONT POINT-SIZE="14" COLOR="#666"> — Finetuned 1ch Climate SR x4</FONT></TD></TR>
            <TR><TD><FONT POINT-SIZE="13" COLOR="#777">embed=96 | depths=[6,6,6,6] | heads=[6,6,6,6] | window=8 | 3ch to 1ch adapted</FONT></TD></TR>
        </TABLE>>
        labelloc=t]
    node [fontname="Helvetica" fontsize=15 shape=box style="filled,rounded"
          penwidth=1.3 margin="0.18,0.08"]
    edge [fontname="Helvetica" fontsize=12 penwidth=1.3 arrowsize=0.8]

    lr [label=<<B>LR Input</B><BR/><FONT POINT-SIZE="12">1 x H x W</FONT>>
        fillcolor="{IO}"]
    ci [label=<<B>Conv 3x3</B><BR/><FONT POINT-SIZE="12">1 to 96</FONT>>
        fillcolor="{CONV}" fontcolor=white]
    r1 [label=<<B>RSTB 1</B><BR/><FONT POINT-SIZE="12">6 x STL</FONT>>
        fillcolor="{ATTN}" fontcolor=white]
    r2 [label=<<B>RSTB 2</B><BR/><FONT POINT-SIZE="12">6 x STL</FONT>>
        fillcolor="{ATTN}" fontcolor=white]
    r3 [label=<<B>RSTB 3</B><BR/><FONT POINT-SIZE="12">6 x STL</FONT>>
        fillcolor="{ATTN}" fontcolor=white]
    r4 [label=<<B>RSTB 4</B><BR/><FONT POINT-SIZE="12">6 x STL</FONT>>
        fillcolor="{ATTN}" fontcolor=white]
    cm [label=<<B>Conv 3x3</B><BR/><FONT POINT-SIZE="12">96 to 96</FONT>>
        fillcolor="{CONV}" fontcolor=white]
    add [label="+" shape=circle fixedsize=true width=0.5 height=0.5
         fillcolor="{SKIP}" fontsize=20 fontcolor="#333"]
    cb [label=<<B>Conv 3x3</B><BR/><FONT POINT-SIZE="12">96 to 64</FONT>>
        fillcolor="{CONV}" fontcolor=white]
    ps [label=<<B>2x PixelShuffle(2)</B><BR/><FONT POINT-SIZE="12">64ch, x4 total</FONT>>
        fillcolor="{SAMP}" fontcolor=white]
    co [label=<<B>Conv 3x3</B><BR/><FONT POINT-SIZE="12">64 to 1</FONT>>
        fillcolor="{CONV}" fontcolor=white]
    hr [label=<<B>HR Output</B><BR/><FONT POINT-SIZE="12">1 x 4H x 4W</FONT>>
        fillcolor="{IO}"]

    lr -> ci -> r1 -> r2 -> r3 -> r4 -> cm -> add -> cb -> ps -> co -> hr
    ci -> add [style=dashed color="{SKIP}" penwidth=2.5 constraint=false
        label=<<FONT COLOR="{SKIP_T}"><B>global residual</B></FONT>>]
}}'''

    # Part 2: RSTB detail
    rstb = f'''digraph RSTB {{
    graph [rankdir=LR fontname="Helvetica" bgcolor=white pad="0.3"
           nodesep=0.3 ranksep=0.35
        label=<<B><FONT POINT-SIZE="20">RSTB</FONT>
        <FONT POINT-SIZE="14" COLOR="#666"> — Residual Swin Transformer Block</FONT></B>>
        labelloc=t]
    node [fontname="Helvetica" fontsize=14 shape=box style="filled,rounded"
          penwidth=1.3 margin="0.15,0.06"]
    edge [fontname="Helvetica" fontsize=11 penwidth=1.2 arrowsize=0.7]

    inp [label="In" shape=plaintext fontsize=14]
    s1 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">W-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    s2 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">SW-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    s3 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">W-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    s4 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">SW-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    s5 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">W-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    s6 [label=<<B>STL</B><BR/><FONT POINT-SIZE="11">SW-MSA</FONT>>
        fillcolor="{RES}" fontcolor=white]
    cv [label=<<B>Conv 3x3</B>> fillcolor="{CONV}" fontcolor=white]
    ad [label="+" shape=circle fixedsize=true width=0.4 height=0.4
        fillcolor="{SKIP}" fontsize=16 fontcolor="#333"]
    out [label="Out" shape=plaintext fontsize=14]

    inp -> s1 -> s2 -> s3 -> s4 -> s5 -> s6 -> cv -> ad -> out
    inp -> ad [style=dashed color="{SKIP}" penwidth=2 constraint=false
        label=<<FONT POINT-SIZE="11" COLOR="{SKIP_T}"><B>residual</B></FONT>>]
}}'''

    # Part 3: STL detail
    stl = f'''digraph STL {{
    graph [rankdir=LR fontname="Helvetica" bgcolor=white pad="0.3"
           nodesep=0.3 ranksep=0.35
        label=<<B><FONT POINT-SIZE="20">STL</FONT>
        <FONT POINT-SIZE="14" COLOR="#666"> — Swin Transformer Layer</FONT></B>>
        labelloc=t]
    node [fontname="Helvetica" fontsize=14 shape=box style="filled,rounded"
          penwidth=1.3 margin="0.15,0.06"]
    edge [fontname="Helvetica" fontsize=11 penwidth=1.2 arrowsize=0.7]

    inp [label="In" shape=plaintext fontsize=14]
    n1 [label="LayerNorm" fillcolor="{NORM}" fontcolor=white]
    msa [label=<<B>W / SW-MSA</B><BR/><FONT POINT-SIZE="11">6 heads, window=8</FONT>>
         fillcolor="{ATTN}" fontcolor=white]
    a1 [label="+" shape=circle fixedsize=true width=0.35 height=0.35
        fillcolor="{SKIP}" fontsize=14 fontcolor="#333"]
    n2 [label="LayerNorm" fillcolor="{NORM}" fontcolor=white]
    mlp [label=<<B>MLP</B><BR/><FONT POINT-SIZE="11">96, 384, 96</FONT>>
         fillcolor="{CONV}" fontcolor=white]
    a2 [label="+" shape=circle fixedsize=true width=0.35 height=0.35
        fillcolor="{SKIP}" fontsize=14 fontcolor="#333"]
    out [label="Out" shape=plaintext fontsize=14]

    inp -> n1 -> msa -> a1 -> n2 -> mlp -> a2 -> out
    inp -> a1 [style=dashed color="{SKIP}" penwidth=1.5 constraint=false
        label=<<FONT POINT-SIZE="10" COLOR="{SKIP_T}"><B>residual</B></FONT>>]
    a1 -> a2 [style=dashed color="{SKIP}" penwidth=1.5 constraint=false
        label=<<FONT POINT-SIZE="10" COLOR="{SKIP_T}"><B>residual</B></FONT>>]
}}'''

    img_main = _render_to_image(main)
    img_rstb = _render_to_image(rstb)
    img_stl = _render_to_image(stl)

    combined = _vstack([img_main, img_rstb, img_stl], gap=30)
    out_path = Path(path)
    combined.save(out_path, dpi=(DPI, DPI))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    gen_flow_unet("figures/arch_flow_unet_eval.png")
    gen_swinir("figures/arch_swinir_eval.png")
