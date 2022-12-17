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
def dummy_func(
    a: float,
    b: float,
    c: float,
):
    def func(
        x: np.ndarray | float,
    ) -> np.ndarray | float:
        return a * x**2 + b * x + c

    return func


@pytest.fixture
def dummy_grad(
    a: float,
    b: float,
    c: float,
):
    def grad(
        x: np.ndarray | float,
    ) -> np.ndarray | float:
        return a * x + b

    return grad


def test_newton(dummy_func: callable, dummy_grad: callable):
    root = newton.find_root(
        func=dummy_func,
        grad=dummy_grad,
        x0=4.0,
        max_num_iterations=100000,
    )
    assert abs(root - 1.0) < 1e-3
