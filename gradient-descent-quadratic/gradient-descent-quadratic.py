def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations of gradient descent
    for f(x) = a*x^2 + b*x + c.
    """
    x = float(x0)
    a = float(a)
    b = float(b)
    lr = float(lr)

    for _ in range(int(steps)):
        grad = 2.0 * a * x + b
        x = x - lr * grad

    return float(x)
