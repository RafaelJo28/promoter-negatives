"""Window features used for matching (shared by Recipe C, Recipe D and the controls).

All features expect a 601-letter window. Lowercase letters mean soft-masked repeats.
Zones are index 0-199, 200-400 and 401-600.
"""

WINDOW_LENGTH = 601
ZONES = ((0, 200), (200, 401), (401, 601))

def _gc(s):
    u = s.upper()
    return (u.count("G") + u.count("C")) / len(u)

def window_features(seq):
    if len(seq) != WINDOW_LENGTH:
        raise ValueError(f"expected {WINDOW_LENGTH} letters, got {len(seq)}")
    u = seq.upper()
    out = {
        "gc": _gc(u),
        "cpg": u.count("CG"),
        "repeat_frac": sum(c.islower() for c in seq) / len(seq),
    }
    for k, (a, b) in enumerate(ZONES, start=1):
        out[f"gc_zone{k}"] = _gc(u[a:b])
    return out
