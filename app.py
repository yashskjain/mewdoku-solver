"""Mewdoku solver web app: upload a board photo, get the cat placement."""

APP_VERSION = "1.0"

import base64
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from solver import solve_meowdoku, check_solution
from vision import detect_and_warp, detect_grid_size, extract_regions

# --- Design tokens: quiet neutrals, one deep ember accent, 4px spacing ---
TOKENS_CSS = """
<style>
:root {
  color-scheme: light dark;
  --brand-50: #FFF7ED;
  --brand-700: #C2410C;
  --brand-800: #9A3412;
  --success: #16A34A;
  --warning: #D97706;
  --error: #DC2626;
  --muted: #6B7280;
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-6: 24px; --space-8: 32px; --space-12: 48px;
  --radius-md: 6px; --radius-lg: 8px; --radius-xl: 12px; --radius-full: 9999px;
  --text-sm: 0.8125rem;  /* 13px */
  --text-md: 0.875rem;   /* 14px */
  --text-lg: 0.9375rem;  /* 15px */
}
.hero-sub { font-size: var(--text-lg); font-weight: 500; color: var(--muted);
  margin: 0 0 var(--space-2); }
.steps { display: flex; gap: var(--space-2); flex-wrap: wrap; margin: var(--space-2) 0 var(--space-6); }
.step { display: inline-flex; align-items: center; gap: var(--space-2);
  padding: 6px 14px; border-radius: var(--radius-full);
  font-size: var(--text-md); font-weight: 600; border: 1px solid transparent; }
.step-done { background: var(--brand-50); color: var(--brand-800); border-color: var(--brand-700); }
.step-current { border-color: var(--brand-700); color: var(--brand-700); }
.step-todo { color: var(--muted); border-color: var(--muted); }
.hint { color: var(--muted); font-size: var(--text-sm); line-height: 1.5; }
.board-frame { aspect-ratio: 1 / 1; overflow: hidden; border-radius: var(--radius-lg);
  background: var(--brand-50); }
.board-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
.stage-empty { aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center;
  border: 1px dashed var(--muted); border-radius: var(--radius-lg);
  color: var(--muted); font-size: var(--text-sm); line-height: 1.5; text-align: center;
  padding: var(--space-4); margin-bottom: var(--space-2); }
.board-cap { color: var(--muted); font-size: var(--text-sm); margin-top: var(--space-2);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.board-cap-ok { color: var(--success); font-weight: 600; }
/* Equal-height panels: bordered cards stretch to the tallest column. */
div[data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }
.stButton > button, .stDownloadButton > button {
  border-radius: var(--radius-lg); font-weight: 600; min-height: 40px;
  touch-action: manipulation; cursor: pointer;
  transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
input:focus-visible { outline: 2px solid var(--brand-700); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
"""

PALETTE = [
    (255, 122, 140), (255, 178, 102), (255, 224, 102), (105, 220, 140),
    (100, 175, 255), (165, 135, 255), (255, 135, 215), (205, 205, 210),
    (232, 165, 95), (75, 200, 200),
]

st.set_page_config(page_title="Mewdoku Solver", page_icon="🐱", layout="wide")
st.markdown(TOKENS_CSS, unsafe_allow_html=True)

st.title("🐱 Mewdoku Solver")
st.markdown(
    '<p class="hero-sub">One cat per row, column and color region — '
    "no two cats touching, even diagonally.</p>",
    unsafe_allow_html=True,
)


def render_board(region_map, cats=(), cell=80):
    """Draw the board: soft region colors, grid lines, cats as orange faces."""
    n = len(region_map)
    img = Image.new("RGB", (n * cell, n * cell), "white")
    d = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            color = PALETTE[region_map[r][c] % len(PALETTE)]
            d.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell], fill=color)
    for i in range(n + 1):
        w = 3 if i in (0, n) else 1
        d.line([(i * cell, 0), (i * cell, n * cell)], fill="black", width=w)
        d.line([(0, i * cell), (n * cell, i * cell)], fill="black", width=w)
    for r, c in cats:
        x, y = c * cell + cell // 2, r * cell + cell // 2
        rad, ear = cell // 3, cell // 5
        d.polygon([(x - rad, y - rad + 6), (x - rad - 2, y - rad - ear), (x - rad // 2, y - rad - 2)],
                  fill="orange", outline="black")
        d.polygon([(x + rad, y - rad + 6), (x + rad + 2, y - rad - ear), (x + rad // 2, y - rad - 2)],
                  fill="orange", outline="black")
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill="orange", outline="black", width=2)
        d.ellipse([x - rad // 2, y - rad // 4, x + rad // 2, y + rad // 2], fill="#ffd9a0")
    return img


def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def board_frame(pil_img, alt):
    """Fixed 1:1 stage box: identical on screen for every board, so panel
    sizes never shift between empty, detected, and solved states."""
    return (f'<div class="board-frame"><img alt="{alt}" decoding="async" '
            f'src="data:image/png;base64,{_b64(pil_img)}" /></div>')


def preview_frame(bgr, max_side=800):
    """Downscaled full photo (aspect kept, CSS crops the box)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s > max_side:
        rgb = cv2.resize(rgb, (int(w * max_side / s), int(h * max_side / s)))
    return board_frame(Image.fromarray(rgb), "Uploaded puzzle photo")


def steps_html(has_photo, has_board, has_solution):
    """Wizard progress indicator: completed, current, upcoming."""
    states = [
        ("Upload", has_photo),
        ("Review", has_board),
        ("Solution", has_solution),
    ]
    current = next((i for i, (_, done) in enumerate(states) if not done), len(states))
    pills = []
    for i, (label, done) in enumerate(states):
        if done:
            cls, mark = "step-done", "✓ "
        elif i == current:
            cls, mark = "step-current", ""
        else:
            cls, mark = "step-todo", ""
        pills.append(f'<span class="step {cls}">{mark}{label}</span>')
    return f'<div class="steps">{"".join(pills)}</div>'


def clear_results():
    """A new (or removed) file invalidates results from the previous one."""
    for key in ("regions", "cats", "detected_n", "photo_bytes", "solve_error"):
        st.session_state.pop(key, None)


def reset():
    clear_results()
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    # No st.rerun(): the button interaction reruns automatically,
    # and rerun calls are not allowed inside callbacks.


def do_detect():
    """Detect grid size + regions from the uploaded photo bytes."""
    raw = st.session_state.get("photo_bytes")
    if not raw:
        return
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    with st.status("Detecting board…", expanded=False) as status:
        st.write("Finding the grid…")
        warped = detect_and_warp(bgr)
        st.write("Counting rows and columns…")
        auto_n = detect_grid_size(warped)
        st.write("Reading the color regions…")
        st.session_state["regions"] = extract_regions(warped, auto_n)
        st.session_state["detected_n"] = auto_n
        st.session_state["cats"] = None
        st.session_state["solve_error"] = None
        status.update(label=f"Board detected ({auto_n}×{auto_n})", state="complete")


regions = st.session_state.get("regions")
detected_n = st.session_state.get("detected_n")
regions_ok = regions is not None and detected_n is not None
cats = st.session_state.get("cats")

# Upload bar: one obvious dropzone up top; previews live in the panels below.
uploaded = st.file_uploader("Board photo", type=["jpg", "jpeg", "png"],
                            help="JPG or PNG photo of a Mewdoku board.",
                            key=f"board_photo_{st.session_state.get('uploader_key', 0)}",
                            on_change=clear_results)
if regions is None and cats is None:
    st.markdown(
        '<p class="hint">Straight-on photo, good light, whole grid visible. '
        "No photo handy? Upload <b>assets/sample_board.png</b> from the repo.</p>",
        unsafe_allow_html=True,
    )

bgr = None
if uploaded is not None:
    st.session_state["photo_bytes"] = uploaded.getvalue()
    try:
        with open("last_upload.png", "wb") as f:
            f.write(uploaded.getvalue())
    except OSError:
        pass
    photo = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(photo, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Couldn't read that file. Please upload a JPG or PNG photo.")

st.markdown(steps_html(uploaded is not None, bool(regions_ok), bool(cats)),
            unsafe_allow_html=True)

col_up, col_mid, col_right = st.columns(3, gap="medium")

with col_up:
    with st.container(border=True):
        st.subheader("Upload")
        if bgr is None:
            st.markdown(
                '<div class="stage-empty">Drop a photo above — '
                "your preview lands here.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(preview_frame(bgr), unsafe_allow_html=True)
            st.markdown('<div class="board-cap">Your photo</div>', unsafe_allow_html=True)

with col_mid:
    with st.container(border=True):
        st.subheader("Review")
        if not regions_ok:
            st.markdown(
                '<div class="stage-empty">Press Detect board below — '
                "the board lands here.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(board_frame(render_board(regions), "Detected board with color regions"),
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="board-cap">Detected {st.session_state["detected_n"]}×{st.session_state["detected_n"]} board</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="hint">Colors look wrong? Retake the photo with even light '
                "and press Detect board again.</p>",
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True):
        st.subheader("Solution")
        if cats and regions_ok:
            st.markdown(board_frame(render_board(regions, cats), "Solved board with cats placed"),
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="board-cap board-cap-ok" role="status">Solved — {len(cats)} cats placed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="stage-empty">Work left to right: upload, detect, solve — '
                "the cats land here.</div>",
                unsafe_allow_html=True,
            )

# --- Actions: one inline bar. Placement never moves; the primary
# highlight follows progress and unavailable actions sit disabled. ---
solved = bool(cats and regions_ok)
if solved:
    buf = io.BytesIO()
    render_board(regions, cats).save(buf, format="PNG")
    dl_data = buf.getvalue()
else:
    dl_data = b""
if st.session_state.get("solve_error"):
    st.error(st.session_state["solve_error"])

bar = st.columns(4, gap="small")
with bar[0]:
    st.button("🔍 Detect board",
              type="primary" if uploaded is not None and not regions_ok else "secondary",
              disabled=uploaded is None, on_click=do_detect, use_container_width=True)
with bar[1]:
    if st.button("🐾 Solve",
                 type="primary" if regions_ok and not cats else "secondary",
                 disabled=not regions_ok, use_container_width=True):
        with st.spinner("Placing the cats…"):
            try:
                solution = solve_meowdoku(regions)
            except ValueError:
                solution = None
        if solution is None or not check_solution(regions, solution)[0]:
            st.session_state["cats"] = None
            st.session_state["solve_error"] = "No solution found. Re-detect from a clearer, straight-on photo."
        else:
            st.session_state["cats"] = solution
            st.session_state["solve_error"] = None
        st.rerun()
with bar[2]:
    st.download_button("⬇️ Download solution", data=dl_data,
                       file_name="mewdoku_solution.png", mime="image/png",
                       disabled=not solved, use_container_width=True)
with bar[3]:
    st.button("↺ Solve another", on_click=reset,
              disabled=uploaded is None and not regions_ok,
              use_container_width=True)

with st.expander("How it works"):
    st.markdown(
        "- **Upload** a photo and press Detect board: the grid size is counted, "
        "the grid straightened and cell colors grouped into regions.\n"
        "- **Review** the detected board against your photo.\n"
        "- **Solve** places exactly one cat per row, column and region, "
        "with no two cats touching — not even diagonally."
    )

st.caption(f"Mewdoku Solver v{APP_VERSION} — open source (MIT).")
