r"""Newton's method

Newton's method, also known as the Newton-Raphson method,
is a root-finding algorithm which produces successively better approximations
to the roots (or zeros) of a real-valued function.

The basic idea for Newton's method is to linearize the function
at a given point $x_0$.

Newton's method is given as the formula as below.

$$
f(x_0) = (x_1 - x_0)f'(x_0)
$$

where, $x_0$ is the linearization point, and $x_1$ is the in the vicinity of $x_0$.

It could be overwritten as

$$
f(x_0) = \delta x f'(x_0).
$$

If we move $x_1$ to the other side of the equation, we get

$$
x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}.
$$

Here, $x_1$ is the place where $f(x)$ decreases.
"""

import numpy as np


def find_root(
    F: callable,
    J: callable,
    x0: np.ndarray,
    max_num_iterations: int = 1000,
    eps: float = 1e-10,
    step_size: float = 1.0,
) -> np.ndarray:
    """Find the root of the given function F

    Args:
        F (callable): target function
        J (callable): derivatives of the target function
        x0 (np.ndarray): initial value of x
        max_num_iterations (int, optional): maximum number of iterations. Defaults to 1000.
        eps (float, optional): stopping criterion. Defaults to 1e-10.
        step_size (float, optional): step size. Defaults to 1.0.

    Returns:
        np.ndarray: zeros of the function F
    """
    x = x0
    for i in range(max_num_iterations):
        if abs(F(x)) < eps:
            # print(f"current x={x}, F(x)={F(x)}.")
            break
        x -= F(x) / J(x) * step_size

    # print(f"F(x)={F(x)}, x={x}, i={i}.")
    if abs(F(x)) > eps:
        raise ValueError(f"Cannot find the root! x={x}.")
    return x


def minimize(
    F: callable,
    J: callable,
    H: callable,
    x0: np.ndarray,
    max_num_iterations: int = 1000,
    eps: float = 1e-10,
    step_size: float = 1.0,
) -> np.ndarray:
    minimizer = find_root(
        F=J,
        J=H,
        x0=x0,
        max_num_iterations=max_num_iterations,
        eps=eps,
        step_size=step_size,
    )
    return minimizer
