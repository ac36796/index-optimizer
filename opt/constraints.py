from abc import ABCMeta
from abc import abstractmethod
from typing import List, Tuple

import numpy as np

__all__ = [
    'ConstraintInterface',
    'FullyInvestConstraint',
    'AccountingConstraint',
    'TurnoverConstraint',
    'IndexDeviateConstraint',
    'TradeNotionalConstraint',
]


class ConstraintInterface(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        return (hasattr(__subclass, 'get_eq_constraint') and
                callable(__subclass.get_eq_constraint) and
                hasattr(__subclass, 'get_neq_constraint') and
                callable(__subclass.get_neq_constraint) or NotImplemented)

    @abstractmethod
    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Get equality constraints to optimizer instance."""
        raise NotImplementedError

    @abstractmethod
    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Get inequality constraints to optimizer instance."""
        raise NotImplementedError


class FullyInvestConstraint(ConstraintInterface):
    """Fully invest all available capital"""

    def __init__(self, tickers: List[str]) -> None:
        self._tickers = tickers

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        n = len(self._tickers)
        return [[np.ones(n).T, np.zeros(n).T, np.zeros(n).T]], [1]

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [], []


class AccountingConstraint(ConstraintInterface):
    """New position should be original position plus buy amount, or minus sell amount.
    """

    def __init__(self, tickers: List[str], x0: np.ndarray) -> None:
        assert len(tickers) == len(x0), ('shape mismatch! '
                                         f'tickers: {len(tickers)}, x0: {len(x0)}')
        self._tickers = tickers
        self._x0 = x0

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        n = len(self._tickers)
        return [[np.eye(n), np.eye(n), -np.eye(n)]], [self._x0.T]

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [], []


class TurnoverConstraint(ConstraintInterface):

    def __init__(self, tickers: List[str], turnover_limit: float) -> None:
        self._tickers = tickers
        self._turnover_limit = turnover_limit

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [], []

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        n = len(self._tickers)
        return [[np.zeros(n).T, np.ones(n).T, np.ones(n).T]], [self._turnover_limit]


class IndexDeviateConstraint(ConstraintInterface):

    def __init__(self, tickers: List[str], idx: np.ndarray, idx_deviate: float) -> None:
        assert len(tickers) == len(idx), (f'shape mismatch! '
                                          f'tickers: {len(tickers)}, idx: {len(idx)}')
        self._tickers = tickers
        self._idx = idx
        self._idx_deviate = idx_deviate

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [], []

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        n = len(self._tickers)
        ub = self._idx * (1 + self._idx_deviate)
        lb = self._idx * (1 - self._idx_deviate)
        return [
            [-np.eye(n), np.zeros((n, n)),
             np.zeros((n, n))],
            [np.eye(n), np.zeros((n, n)), np.zeros((n, n))],
        ], [-lb, ub]


class TradeNotionalConstraint(ConstraintInterface):
    """Trade notional should all be positive."""

    def __init__(self, tickers: List[str]) -> None:
        self._tickers = tickers

    def get_eq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return [], []

    def get_neq_constraint(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        n = len(self._tickers)
        return [
            [np.zeros((n, n)), -np.eye(n),
             np.zeros((n, n))],
            [np.zeros((n, n)), np.zeros((n, n)), -np.eye(n)],
        ], [np.zeros(n), np.zeros(n)]
