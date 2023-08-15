from typing import List, Tuple

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from opt.constraints import ConstraintInterface
from opt.optimizer import Optimizer

tickers = ['A', 'B']
cov = np.eye(2)
x0 = np.array([100., 1000.])
signal = np.array([0.01, 0.1])


def test_update_and_reset():

    o = Optimizer(tickers=tickers, cov=cov)

    assert_array_almost_equal(
        [
            [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros(
                (2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros(
                (2, 2)), np.zeros((2, 2))],
        ],
        o.P,
    )
    assert all(expected == res for expected, res in zip(tickers, o.tickers))

    o.update_x0(x0).update_signal(signal)
    assert_array_almost_equal(x0, o.x0)
    assert_array_almost_equal(signal, o.signal)
    assert_array_almost_equal(
        [np.array([-0.01, -0.1]), np.zeros(2),
         np.zeros(2)],
        o.q,
    )

    o.reset()
    assert_array_almost_equal([], o.x0)
    assert_array_almost_equal([], o.signal)
    assert_array_almost_equal([], o.q)
    assert_array_almost_equal(
        [
            [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros(
                (2, 2)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros(
                (2, 2)), np.zeros((2, 2))],
        ],
        o.P,
    )


class FakeConstraint(ConstraintInterface):

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [[np.eye(2), np.eye(2), -np.eye(2)]], [x0.T]

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [
            [np.zeros((2, 2)), -np.eye(2),
             np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), -np.eye(2)],
        ], [np.zeros(2), np.zeros(2)]


@pytest.fixture
def cstr() -> FakeConstraint:
    return FakeConstraint()


def test_add_constraints(cstr):
    o = Optimizer(tickers=tickers, cov=cov)

    o.add_constraints([cstr, cstr])
    assert_array_almost_equal(
        [
            [np.eye(2), np.eye(2), -np.eye(2)],
            [np.eye(2), np.eye(2), -np.eye(2)],
        ],
        o.A,
    )
    assert_array_almost_equal(
        [
            [np.zeros(
                (2, 2)), -np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), -np.eye(2)],
            [np.zeros(
                (2, 2)), -np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), -np.eye(2)],
        ],
        o.G,
    )
    assert_array_almost_equal([x0, x0], o.b)
    assert_array_almost_equal([np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)], o.h)


def test_solver(cstr):
    o = Optimizer(tickers=tickers, cov=cov)
    res = o.reset().update_x0(x0).update_signal(signal).add_constraints([cstr]).solve()

    assert_array_almost_equal(
        [0, 0, 395, 1004, 295, 4],
        res,
        decimal=0,
    )
