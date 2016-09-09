"""Tests for main.py"""
from ..main import _solve_weight_vector
import numpy as np
from nose.tools import assert_raises, assert_equal, assert_true


def test_solve_weight_vector():
    # smoke test
    n_nodes = 4
    n_communities = 2
    n_similarities = 3
    delta = 0.3
    similarities = np.random.random((n_similarities,
                                     n_nodes * (n_nodes-1)/2)) * 10
    grouping_matrix = np.random.random((n_nodes, n_communities))

    weight = _solve_weight_vector(similarities, grouping_matrix, delta)
    assert_equal(weight.ndim, 2)
    assert_equal(weight.shape[1], n_similarities)
    assert_true(np.all(weight >= 0))

    # check raises
    assert_raises(ValueError,
                  _solve_weight_vector, similarities, grouping_matrix, -1)
    similarities_invalid = similarities.copy()
    similarities_invalid[0, 3] = -4.
    assert_raises(ValueError,
                  _solve_weight_vector, similarities_invalid,
                  grouping_matrix, delta)
