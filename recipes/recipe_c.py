"""Recipe C: pair-preserving block shuffle.

Each window is cut into blocks of about 100 letters. Inside every block the letters are
shuffled with the Altschul-Erickson Eulerian-path algorithm, which keeps all 16
neighbouring-letter-pair counts and the first and last letter of the block. The lowercase
(repeat) mask of the original window is kept in place.
"""
from collections import defaultdict

def block_bounds(length, block_size=100):
    n_blocks = max(1, length // block_size)
    bounds = [(i * block_size, (i + 1) * block_size) for i in range(n_blocks)]
    bounds[-1] = (bounds[-1][0], length)  # last block takes the remainder
    return bounds

def _reaches(v, last, last_exit):
    cur = v
    for _ in range(len(last_exit) + 1):
        if cur == last:
            return True
        cur = last_exit.get(cur)
        if cur is None:
            return False
    return cur == last

def eulerian_shuffle(seq, rng):
    """Uniform random sequence with the same first letter and the same pair counts as seq."""
    n = len(seq)
    if n < 3:
        return seq
    last = seq[-1]
    edges = defaultdict(list)  # letter -> list of letters that follow it
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    verts = [v for v in edges if v != last]
    while True:  # last-exit edges must form a tree pointing at the last letter
        last_exit = {v: rng.choice(edges[v]) for v in verts}
        if all(_reaches(v, last, last_exit) for v in verts):
            break
    out = {}
    for v, nxt in edges.items():
        lst = list(nxt)
        if v in last_exit:
            lst.remove(last_exit[v])
            rng.shuffle(lst)
            lst.append(last_exit[v])
        else:
            rng.shuffle(lst)
        out[v] = iter(lst)
    cur = seq[0]
    res = [cur]
    for _ in range(n - 1):
        cur = next(out[cur])
        res.append(cur)
    return "".join(res)

def make_negative(window, rng, block_size=100):
    """Recipe C negative for one case-preserving window."""
    upper = window.upper()
    shuffled = "".join(
        eulerian_shuffle(upper[a:b], rng) for a, b in block_bounds(len(window), block_size)
    )
    return "".join(ch.lower() if orig.islower() else ch for ch, orig in zip(shuffled, window))
