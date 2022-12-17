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
    func: callable,
    grad: callable,
    x0: np.ndarray,
    max_num_iterations: int = 1000,
    eps: float = 1e-10,
    step_size: float = 1.0,
):
    x = x0
    values = []
    for i in range(max_num_iterations):
        if abs(func(x)) < eps:
            break
        x -= func(x) / grad(x) * step_size
        values.append(x)

    return x
