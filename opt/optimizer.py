from __future__ import annotations

from typing import List, Optional

from cvxopt import matrix
from cvxopt import solvers
import numpy as np

from opt.constraints import ConstraintInterface

__all__ = [
    'Optimizer',
]

solvers.options['show_progress'] = False


class Optimizer:

    def __init__(self, tickers: List[str], cov: np.ndarray) -> None:

        assert len(tickers) == len(cov), (f'shape mismatch!'
                                          f'tickers: {len(tickers)}, cov: {cov.shape}')
        self._tickers = tickers
        self._cov = cov

        # initial position
        self._x0: Optional[np.ndarray] = None
        self._signal: Optional[np.ndarray] = None
        self._P: List[List[np.ndarray]] = None
        self._A: List[List[np.ndarray]] = None
        self._G: List[List[np.ndarray]] = None
        self._q: List[np.ndarray] = None
        self._b: List[np.ndarray] = None
        self._h: List[np.ndarray] = None

    def _init_covariance(self) -> None:
        n = len(self._tickers)
        self._P = [
            [self._cov, np.zeros((n, n)), np.zeros((n, n))],
            [np.zeros((n, n)), np.zeros((n, n)),
             np.zeros((n, n))],
            [np.zeros((n, n)), np.zeros((n, n)),
             np.zeros((n, n))],
        ]

    def update_x0(self, x0: np.ndarray) -> Optimizer:
        self._x0 = x0
        return self

    def update_signal(self, signal: np.ndarray) -> Optimizer:
        self._signal = signal

        n = len(self._tickers)
        self._q = [-signal, np.zeros(n), np.zeros(n)]
        return self

    def add_constraints(self, constraints: List[ConstraintInterface]) -> Optimizer:
        for cstr in constraints:
            a, b = cstr.get_eq_constraints()
            self._A += a
            self._b += b

            g, h = cstr.get_neq_constraint()
            self._G += g
            self._h += h
        return self

    def reset(self) -> None:
        self._x0 = None
        self._signal = None
        self._A = None
        self._G = None
        self._q = None
        self._b = None
        self._h = None

    def solve(self) -> np.ndarray:
        P, q = matrix(np.block(self._P)), matrix(np.block(self._q))
        A, b = matrix(np.block(self._A)), matrix(np.block(self._b))
        G, h = matrix(np.block(self._G)), matrix(np.block(self._h))

        res = solvers.qp(P, q, G=G, h=h, A=A, b=b)

        ret = self._x0
        # TODO: handle the case when optimization fails
        if res['status'] == 'optimal':
            ret = np.array(res['x']).ravel()
        return ret
