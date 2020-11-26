import numpy
from DSK import DSK_algorithm
from HalfDivideMethod import half_divide

global a


def func(x, y):
    return numpy.sqrt(2 * x ** 2 + y ** 2 + 1) - x - y + numpy.exp(x ** 2 + 2 * y ** 2)


def alpha(a):
    x_a = x - grad(x, y)[0] * a
    y_a = y - grad(x, y)[1] * a

    return func(x_a, y_a)


def grad(x, y):
    diffX = -1 + 2 * numpy.exp(x ** 2 + 2 * y ** 2) * x + (2 * x) / numpy.sqrt(1 + 2 * x ** 2 + y ** 2)
    diffY = -1 + 4 * numpy.exp(x ** 2 + 2 * y ** 2) * y + y / numpy.sqrt(1 + 2 * x ** 2 + y ** 2)
    grad = []
    grad.append(diffX)
    grad.append(diffY)
    return grad


# Input data
x = 0
y = 0
h = 2
a0 = 0.1
eps = 10**(-5)
# Gradient method
print("Iter", "                  Result                  ", "      interval", "        Half divide method")

xk = []
xk.append(x)
xk.append(y)
k = 0
while True:

    x = xk[0]
    y = xk[1]
    gradient = grad(x, y)
    norma = numpy.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    if norma <= eps:
        print(xk)
        break

    if norma > eps:
        min = half_divide(alpha, DSK_algorithm(alpha, a0, h), eps)
        xk[0] = x - gradient[0] * min
        xk[1] = y - gradient[1] * min
    k = k + 1;

    print(k, xk, DSK_algorithm(alpha, a0, h), min)
    if not numpy.sqrt((xk[0] - x) ** 2 + (xk[1] - y) ** 2) > eps:
        break
