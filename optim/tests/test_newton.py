from optim import newton
import numpy as np
import pytest


@pytest.fixture
def a() -> float:
    return 1.0


@pytest.fixture
def b() -> float:
    return -2.0


@pytest.fixture
def c() -> float:
    return 1.0


@pytest.fixture
def dummy_F(
    a: float,
    b: float,
    c: float,
):
    def F(
        x: np.ndarray | float,
    ) -> np.ndarray | float:
        return a * x**2 + b * x + c

    return F


@pytest.fixture
def dummy_J(
    a: float,
    b: float,
):
    def J(
        x: np.ndarray | float,
    ) -> np.ndarray | float:
        return 2.0 * a * x + b

    return J


@pytest.fixture
def dummy_H(
    a: float,
):
    def H(
        x: np.ndarray | float,
    ) -> np.ndarray | float:
        return 2.0 * a

    return H


def test_newton_root(dummy_F: callable, dummy_J: callable):
    eps = 1e-5
    root = newton.find_root(
        F=dummy_F,
        J=dummy_J,
        x0=4.0,
        eps=eps,
        max_num_iterations=1000,
    )
    assert abs(root - 1.0) < 0.003, f"error={abs(root-1.0)}."


def test_newton_minimize(dummy_F: callable, dummy_J: callable, dummy_H: callable):
    eps = 1e-5
    x = newton.minimize(
        F=dummy_F,
        J=dummy_J,
        H=dummy_H,
        x0=4.0,
        eps=eps,
    )
    assert abs(x - 1.0) < eps
