import numpy as np

from hansgpt_research.diagnostics import (
    IDS_OPERATORS,
    STRUCTURE_NAMES,
    _normalise_features,
    structure_class,
)


def test_structure_class_recognises_ids_operators() -> None:
    assert structure_class("⿰氵青", "清") == IDS_OPERATORS.index("⿰") + 1
    assert structure_class("清", "清") == 0
    assert structure_class("unknown", "清") == len(STRUCTURE_NAMES) - 1


def test_feature_normalisation_uses_only_training_indices() -> None:
    features = np.asarray([[0.0, 2.0], [2.0, 4.0], [100.0, 100.0]])
    normalised, mean, standard_deviation = _normalise_features(features, np.asarray([0, 1]))
    np.testing.assert_allclose(mean, [[1.0, 3.0]])
    np.testing.assert_allclose(standard_deviation, [[1.0, 1.0]])
    np.testing.assert_allclose(normalised[:2].mean(axis=0), [0.0, 0.0])
