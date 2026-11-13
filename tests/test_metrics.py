import numpy as np
import pytest

from recipes.metrics import DINUCS, kmer_table, smd

def test_smd_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert smd(a, a.copy()) == 0.0

def test_smd_known_value():
    a = np.array([0.0, 1.0, 2.0])
    assert smd(a + 1.0, a) == pytest.approx(1.0)

def test_kmer_table_dinucleotides():
    X = kmer_table(["ACGTACGT"], DINUCS)
    assert X.sum() == pytest.approx(1.0)
    assert X[0, DINUCS.index("AC")] == pytest.approx(2 / 7)
    assert X[0, DINUCS.index("TA")] == pytest.approx(1 / 7)
