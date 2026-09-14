"""Image -> region map extraction for Meowdoku boards."""
import cv2
import numpy as np


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _valid_quad(rect, w, h):
    """Reject sheared/garbage quads: a straight-on board photo warps to a
    near-rectangle. Lets bad contours fall through to the safe center crop."""
    area = cv2.contourArea(rect)
    if area < 0.25 * w * h:
        return False
    if cv2.contourArea(cv2.convexHull(rect)) - area > 0.03 * area:
        return False
    sides = [np.linalg.norm(rect[(i + 1) % 4] - rect[i]) for i in range(4)]
    if max(sides) / (min(sides) + 1e-6) > 1.6:
        return False
    for i in range(4):
        v1 = rect[i] - rect[(i - 1) % 4]
        v2 = rect[(i + 1) % 4] - rect[i]
        cosang = abs(float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        if cosang > 0.5:
            return False
    top_cx = (rect[0][0] + rect[1][0]) / 2
    bot_cx = (rect[2][0] + rect[3][0]) / 2
    left_cy = (rect[0][1] + rect[3][1]) / 2
    right_cy = (rect[1][1] + rect[2][1]) / 2
    if abs(top_cx - bot_cx) > 0.05 * w or abs(left_cy - right_cy) > 0.05 * h:
        return False
    return True


def _trim_mat(bgr, white_level=232, white_frac=0.85, max_dark_run=8, max_trim_frac=0.25):
    """Cut the frame + white mat board prints have around the tiles.

    From each edge, skips a thin dark run (frame, bounded so tall dark
    tile rows never match) followed by a near-white run (mat, bounded).
    Stops at the first tile row. Pitch is unaffected by cropping, while
    leftover margins shrink apparent pitch and break size detection.
    Returns the input unchanged when nothing trimmable is found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white = gray >= white_level
    h, w = gray.shape
    row_wf = white.mean(axis=1)
    col_wf = white.mean(axis=0)

    def lead_trim(fracs):
        n = len(fracs)
        lim = int(n * max_trim_frac)
        i = 0
        j = i
        while j < lim and fracs[j] < 1 - white_frac:
            j += 1
        if j - i <= max_dark_run:
            i = j
        while i < lim and fracs[i] > white_frac:
            i += 1
        return i

    top = lead_trim(row_wf)
    bottom = lead_trim(row_wf[::-1])
    left = lead_trim(col_wf)
    right = lead_trim(col_wf[::-1])
    if top + bottom >= h // 2 or left + right >= w // 2:
        return bgr
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return bgr
    return bgr[top:h - bottom, left:w - right]


def _trim_white_margin(bgr, thresh=245, min_frac=0.02, pad=2):
    """Crop near-white page margins so the board fills the frame.
    Gives up (returns input) when no plausible crop exists."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < thresh
    rows = np.where(mask.mean(axis=1) > min_frac)[0]
    cols = np.where(mask.mean(axis=0) > min_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return bgr
    y0, y1 = max(0, rows[0] - pad), min(bgr.shape[0], rows[-1] + pad + 1)
    x0, x1 = max(0, cols[0] - pad), min(bgr.shape[1], cols[-1] + pad + 1)
    if (y1 - y0) < 0.2 * bgr.shape[0] or (x1 - x0) < 0.2 * bgr.shape[1]:
        return bgr
    return bgr[y0:y1, x0:x1]


def detect_and_warp(bgr, out_size=600):
    """Find the board quad and warp to square.

    Falls back to trimming white page margins + center square crop, so
    straight-on shots still fill the frame when no clean contour exists.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(15, (min(h0, w0) // 10) | 1)  # scale-aware window, odd
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 9)
    k = max(3, min(h0, w0) // 60)
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            # refine sloppy corners to sub-pixel accuracy (bounded 10px nudge)
            gray_fp = gray.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray_fp, approx.reshape(-1, 1, 2).astype(np.float32),
                                       (10, 10), (-1, -1), criteria)
            rect = _order_points(corners.reshape(4, 2))
            if _valid_quad(rect, w0, h0) and cv2.contourArea(cnt) > 0.05 * w0 * h0:
                dst = np.array([[0, 0], [out_size - 1, 0],
                                [out_size - 1, out_size - 1], [0, out_size - 1]], dtype="float32")
                m = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(bgr, m, (out_size, out_size))
                return _reframe(warped, out_size)
    # fallback: trim white margins, then center square crop
    trimmed = _trim_white_margin(bgr)
    h, w = trimmed.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = trimmed[y0:y0 + s, x0:x0 + s]
    warped = cv2.resize(crop, (out_size, out_size))
    return _reframe(warped, out_size)


def _reframe(warped, out_size):
    """Cut frame/mat margins so tiles fill the frame, restore output size."""
    framed = _trim_white_margin(_trim_mat(warped))
    if framed.shape[0] == out_size and framed.shape[1] == out_size:
        return framed
    return cv2.resize(framed, (out_size, out_size))


def _moving_avg(x, w):
    """Edge-safe moving average via cumulative sums."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo = np.clip(idx - w // 2, 0, len(x) - 1)
    hi = np.clip(idx + w // 2 + 1, 1, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _period_score(detail, lag):
    """Normalized autocorrelation at (integer) lag; 1.0 = perfect repetition."""
    x = detail - detail.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    k = int(round(lag))
    if k <= 0 or k >= len(x):
        return 0.0
    return float(np.dot(x[:len(x) - k], x[k:]) / denom)


def _edge_profile(gray, axis):
    """Mean gradient magnitude across lines: grid lines and gutters both
    produce strong edges regardless of polarity or thickness."""
    g = np.abs(np.diff(gray, axis=0 if axis == 1 else 1))
    return g.mean(axis=axis)


def _detect_axis_n(profiles, min_n=5, max_n=10, min_score=0.25):
    """Best grid size for one axis, or None if nothing repeats convincingly."""
    best_n, best_score = None, min_score
    for n in range(min_n, max_n + 1):
        s = 0.0
        for p in profiles:
            det = p - _moving_avg(p, max(8, len(p) // 4))
            s = max(s, _period_score(det, len(p) / n))
        if s > best_score:
            best_n, best_score = n, s
    return best_n, best_score


def detect_grid_size(warped_bgr, min_n=5, max_n=10, default=6):
    """Detect board size N by cell periodicity (works for grid lines and
    gutter-separated tiles alike). Falls back to `default` when unsure."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    row_profiles = [gray.mean(axis=1), _edge_profile(gray, axis=1)]
    col_profiles = [gray.mean(axis=0), _edge_profile(gray, axis=0)]
    n_rows, score_rows = _detect_axis_n(row_profiles, min_n, max_n)
    n_cols, score_cols = _detect_axis_n(col_profiles, min_n, max_n)
    if n_rows is not None and n_rows == n_cols:
        return n_rows
    if n_rows is not None and n_cols is not None:
        return n_rows if score_rows >= score_cols else n_cols
    if n_rows is not None:
        return n_rows
    if n_cols is not None:
        return n_cols
    return default


def extract_regions(warped_bgr, n):
    """Cluster cell-center colors into n regions. Returns the region map."""
    size = warped_bgr.shape[0]
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    cell = size // n
    margin = int(cell * 0.25)  # ignore grid lines at edges
    feats = []
    for r in range(n):
        for c in range(n):
            y0, y1 = r * cell + margin, (r + 1) * cell - margin
            x0, x1 = c * cell + margin, (c + 1) * cell - margin
            patch = lab[y0:y1, x0:x1]
            feats.append(patch.reshape(-1, 3).mean(axis=0))
    feats = np.array(feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, _ = cv2.kmeans(feats, n, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    region_map = labels.reshape(n, n).tolist()
    return region_map
