import pytest

from recipes.features import window_features

def test_zone_gc_and_repeat():
    seq = "A" * 200 + "g" * 201 + "T" * 200
    f = window_features(seq)
    assert f["gc_zone1"] == 0.0
    assert f["gc_zone2"] == 1.0
    assert f["gc_zone3"] == 0.0
    assert f["gc"] == pytest.approx(201 / 601)
    assert f["repeat_frac"] == pytest.approx(201 / 601)
    assert f["cpg"] == 0

def test_cpg_counts_case_insensitive():
    seq = "ACGT" * 150 + "A"  # 601 letters
    assert window_features(seq)["cpg"] == 150
    assert window_features(seq.lower())["cpg"] == 150

def test_wrong_length_raises():
    with pytest.raises(ValueError):
        window_features("ACGT")
