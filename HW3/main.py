import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def draw_plot(xs: np.ndarray, f: Callable, k: int = 1) -> None:
    # Draw 2 plots:
    # On the left subplot, show:
    # - the function f(x) as contours
    # - the values of x as we search for the minimizer
    # On the right subplot, show:
    # - value of f(x) vs. current xi at each iteration of the optimization

    plt.subplots(1, 2)

    plt.subplot(1, 2, 2)
    z = f(xs)
    plt.plot(np.arange(len(z)), z)
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.gca().set_yscale('log')

    plt.subplot(1, 2, 1)

    # Optimization
    plt.plot(xs[:, 0], xs[:, 1], '-x')

    x_low = -2
    x_high = 2
    y_low = -2
    y_high = 2

    # Contour Plot
    num_xs = 20
    num_ys = 20
    x_vals = np.linspace(x_low, x_high, num_xs)
    y_vals = np.linspace(y_low, y_high, num_ys)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy_pairs = np.vstack([X.ravel(), Y.ravel()]).T
    z = f(xy_pairs, k=k)
    Z = z.reshape(num_xs, num_ys)
    plt.gca().contour(X, Y, Z)

    plt.xlim([x_low, x_high])
    plt.ylim([y_low, y_high])
    plt.show()


def f(x: np.ndarray, k: int = 1) -> float:
    # f(x) = x^2 - xy + kx^2
    print(x[1])
    if x.ndim == 1:
        return x[0] ** 2 - x[0] * x[1] + k * x[1] ** 2
    elif x.ndim == 2:
        return x[:, 0] ** 2 - x[:, 0] * x[:, 1] + k * x[:, 1] ** 2


# return the gradient of f(x) and k
def gradf(x: np.ndarray, k: int = 1) -> np.ndarray:
    # grad f(x) = [2x - y + 2kx, -x + 2ky]
    return np.array([2 * x[0] - x[1] + 2 * k * x[0], -x[0] + 2 * k * x[1]])


# return the hessian of f(x) and k
def hessf(x: np.ndarray, k: int = 1) -> np.ndarray:
    # hess f(x) = [[2 + 2k, -1], [-1, 2k]]
    return np.array([[2 + 2 * k, -1], [-1, 2 * k]])


# Implement the gradient descent algorithm
def gradient_descent(
        f: Callable,
        gradf: Callable,
        x0: np.ndarray,
        c: float,
        tau: float,
        epsilon: float,
        plot: bool = False,
        k: int = 1,
) -> np.ndarray:
    xs = [x0]
    while True:
        x = xs[-1]
        grad = gradf(x, k=k)
        if np.linalg.norm(grad) < epsilon:
            break
        xs.append(x - c * grad)
        c *= tau
    xs = np.array(xs)
    if plot:
        draw_plot(xs, f, k=k)
    return xs


x0 = np.array([1., 1.])
c = 0.5
tau = 0.5
epsilon = 0.001
xs1 = gradient_descent(f, gradf, x0, c, tau, epsilon, k=1, plot=True)
xs10 = gradient_descent(f, gradf, x0, c, tau, epsilon, k=10, plot=True)
xs100 = gradient_descent(f, gradf, x0, c, tau, epsilon, k=100, plot=True)
xs1000 = gradient_descent(f, gradf, x0, c, tau, epsilon, k=1000, plot=True)


def newton(
        f: Callable,
        gradf: Callable,
        hessf: Callable,
        x0: np.ndarray,
        epsilon: float,
        plot: bool = False,
        k: int = 1
) -> np.ndarray:
    xs = [x0]
    while True:
        x = xs[-1]
        grad = gradf(x, k=k)
        hess = hessf(x, k=k)
        if np.linalg.norm(grad) < epsilon:
            break
        xs.append(x - np.linalg.inv(hess) @ grad)
    xs = np.array(xs)
    if plot:
        draw_plot(xs, f, k=k)
    return xs
