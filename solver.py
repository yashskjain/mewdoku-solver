"""Meowdoku (Queens-variant) solver.

Rules:
- NxN grid, N colored regions (ids 0..N-1)
- Exactly one cat per row, per column, per region
- No two cats touch, even diagonally
"""
import sys

sys.setrecursionlimit(10000)

def validate_board(regions):
    n = len(regions)
    if n == 0:
        raise ValueError("Empty board")
    for row in regions:
        if len(row) != n:
            raise ValueError("Board must be square (NxN)")
    flat = [c for row in regions for c in row]
    if len(set(flat)) != n:
        raise ValueError(f"Must have exactly N={n} distinct regions, got {len(set(flat))}")


def check_solution(regions, cats):
    """cats: list of (r, c). Returns (ok, msg)."""
    n = len(regions)
    if cats is None or len(cats) != n:
        return False, f"Need {n} cats, got {0 if cats is None else len(cats)}"
    rows = [r for r, _ in cats]
    cols = [c for _, c in cats]
    if len(set(rows)) != n:
        return False, "Two cats share a row"
    if len(set(cols)) != n:
        return False, "Two cats share a column"
    seen_regions = set()
    for r, c in cats:
        if not (0 <= r < n and 0 <= c < n):
            return False, f"Cat out of bounds: {(r, c)}"
        rg = regions[r][c]
        if rg in seen_regions:
            return False, f"Two cats in region {rg}"
        seen_regions.add(rg)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(cats[i][0] - cats[j][0]) <= 1 and abs(cats[i][1] - cats[j][1]) <= 1:
                return False, f"Cats touch: {cats[i]} and {cats[j]}"
    return True, "Solved!"


def solve_meowdoku(regions):
    """Solve from region map. Returns list[(r,c)] or None."""
    validate_board(regions)
    n = len(regions)

    # MRV search over rows; plenty fast for N<=10.
    unplaced_rows = list(range(n))
    col_of_row = [-1] * n
    cols_used = set()
    regions_used = set()
    placed = []  # list of (r, c)

    # Precompute region id per cell (already ints, normalize to 0..N-1)
    uniq = sorted(set(c for row in regions for c in row))
    remap = {v: i for i, v in enumerate(uniq)}
    norm = [[remap[c] for c in row] for row in regions]

    def touches(r1, c1, r2, c2):
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

    def candidates(row):
        out = []
        for c in range(n):
            if c in cols_used:
                continue
            if norm[row][c] in regions_used:
                continue
            ok = True
            for pr, pc in placed:
                if touches(row, c, pr, pc):
                    ok = False
                    break
            if ok:
                out.append(c)
        return out

    # MRV: pick unplaced row with fewest candidates
    def dfs():
        if len(placed) == n:
            return True
        # choose row
        best_row, best_cands = -1, None
        for r in unplaced_rows:
            if col_of_row[r] != -1:
                continue
            cands = candidates(r)
            if not cands:
                return False
            if best_cands is None or len(cands) < len(best_cands):
                best_cands, best_row = cands, r
                if len(best_cands) == 1:
                    break
        r = best_row
        # try rarest-region columns first (helps a bit)
        for c in best_cands:
            col_of_row[r] = c
            cols_used.add(c)
            regions_used.add(norm[r][c])
            placed.append((r, c))
            if dfs():
                return True
            placed.pop()
            regions_used.discard(norm[r][c])
            cols_used.discard(c)
            col_of_row[r] = -1
        return False

    if dfs():
        return sorted(placed)
    return None
