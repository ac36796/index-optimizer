import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from opt.constraints import AccountingConstraint
from opt.constraints import FullyInvestConstraint
from opt.constraints import IndexDeviateConstraint
from opt.constraints import TradeNotionalConstraint
from opt.constraints import TurnoverConstraint

tickers = ['A', 'B']


def test_fully_invest_constraint():
    fi = FullyInvestConstraint(tickers=tickers)

    x, y = fi.get_eq_constraint()
    assert len(x) == len(y) == 1
    x0, x1, x2 = x[0]
    assert_array_equal(np.ones(2).T, x0)
    assert_array_equal(np.zeros(2).T, x1)
    assert_array_equal(np.zeros(2).T, x2)
    assert 1 == y[0]

    x, y = fi.get_neq_constraint()
    assert_array_equal([], x)
    assert_array_equal([], y)


def test_accounting_constraint():
    x0 = np.array([100, 1000])
    a = AccountingConstraint(tickers=tickers, x0=x0)

    x, y = a.get_eq_constraint()
    assert len(x) == len(y) == 1
    x0, x1, x2 = x[0]
    assert_array_equal(np.eye(2), x0)
    assert_array_equal(np.eye(2), x1)
    assert_array_equal(-np.eye(2), x2)
    assert_array_equal([100, 1000], y[0])


def test_turnover_constraint():
    turnover_limit = 0.1
    t = TurnoverConstraint(tickers=tickers, turnover_limit=turnover_limit)

    x, y = t.get_eq_constraint()
    assert_array_equal([], x)
    assert_array_equal([], y)

    x, y = t.get_neq_constraint()
    assert len(x) == len(y) == 1
    x0, x1, x2 = x[0]
    assert_array_equal(np.zeros(2).T, x0)
    assert_array_equal(np.ones(2).T, x1)
    assert_array_equal(np.ones(2).T, x2)
    assert 0.1 == y[0]


def test_index_deviate_constraint():
    idx_deviate = 0.1
    idx = np.array([100, 1000])
    id_cstr = IndexDeviateConstraint(tickers=tickers, idx=idx, idx_deviate=idx_deviate)

    x, y = id_cstr.get_eq_constraint()
    assert_array_equal([], x)
    assert_array_equal([], y)

    x, y = id_cstr.get_neq_constraint()
    assert len(x) == len(y) == 2
    x0, x1, x2 = x[0]
    assert_array_equal(-np.eye(2), x0)
    assert_array_equal(np.zeros((2, 2)), x1)
    assert_array_equal(np.zeros((2, 2)), x2)
    x0, x1, x2 = x[1]
    assert_array_equal(np.eye(2), x0)
    assert_array_equal(np.zeros((2, 2)), x1)
    assert_array_equal(np.zeros((2, 2)), x2)
    y0, y1 = y
    assert_array_almost_equal([-90, -900], y0)
    assert_array_almost_equal([110, 1100], y1)


def test_trade_notional_constraint():
    tn = TradeNotionalConstraint(tickers=tickers)

    x, y = tn.get_neq_constraint()
    assert len(x) == len(y) == 2
    x0, x1, x2 = x[0]
    assert_array_equal(np.zeros((2, 2)), x0)
    assert_array_equal(-np.eye(2), x1)
    assert_array_equal(np.zeros((2, 2)), x2)
    y0, y1 = y
    assert_array_equal(np.zeros(2), y0)
    assert_array_equal(np.zeros(2), y1)

    x, y = tn.get_eq_constraint()
    assert_array_equal([], x)
    assert_array_equal([], y)
