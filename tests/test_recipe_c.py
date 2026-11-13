import itertools
import random
from collections import Counter

from recipes.recipe_c import block_bounds, eulerian_shuffle, make_negative

def pair_counts(s):
    s = s.upper()
    return Counter(s[i:i + 2] for i in range(len(s) - 1))

def rand_seq(rng, n, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(n))

def test_block_bounds_601():
    b = block_bounds(601, 100)
    assert len(b) == 6
    assert b[0] == (0, 100) and b[-2] == (400, 500) and b[-1] == (500, 601)
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))

def test_shuffle_preserves_pairs_and_endpoints():
    rng = random.Random(1)
    for n in (2, 3, 10, 50, 100, 101):
        for _ in range(50):
            s = rand_seq(rng, n)
            t = eulerian_shuffle(s, rng)
            assert len(t) == len(s)
            assert t[0] == s[0] and t[-1] == s[-1]
            assert pair_counts(t) == pair_counts(s)

def test_shuffle_is_uniform_on_small_case():
    s = "AABCABCCA"
    target = pair_counts(s)
    support = set()
    for perm in set(itertools.permutations(s)):
        t = "".join(perm)
        if t[0] == s[0] and pair_counts(t) == target:
            support.add(t)
    assert len(support) >= 3
    rng = random.Random(7)
    n = 30000
    counts = Counter(eulerian_shuffle(s, rng) for _ in range(n))
    assert set(counts) == support
    expected = n / len(support)
    for c in counts.values():
        assert abs(c - expected) < 6 * expected ** 0.5

def test_make_negative_preserves_composition_pairs_and_mask():
    rng = random.Random(3)
    base = rand_seq(rng, 601)
    mask = [rng.random() < 0.12 for _ in range(601)]
    window = "".join(c.lower() if m else c for c, m in zip(base, mask))
    neg = make_negative(window, random.Random(11))
    assert len(neg) == 601
    assert pair_counts(neg) == pair_counts(window)
    assert Counter(neg.upper()) == Counter(window.upper())
    assert [c.islower() for c in neg] == [c.islower() for c in window]
    for a, b in block_bounds(601, 100):
        assert pair_counts(neg[a:b]) == pair_counts(window[a:b])
    assert neg.upper() != window.upper()

def test_reproducible_and_seed_dependent():
    base = rand_seq(random.Random(5), 601)
    a = make_negative(base, random.Random("x"))
    b = make_negative(base, random.Random("x"))
    c = make_negative(base, random.Random("y"))
    assert a == b
    assert a != c
