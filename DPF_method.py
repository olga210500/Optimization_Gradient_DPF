import numpy as np
from DSK import DSK_algorithm
from HalfDivideMethod import half_divide
np.seterr(divide='ignore', invalid='ignore')
global a


def f(x, y):

    return np.sqrt(2 * x ** 2 + y ** 2 + 1) - x - y + np.exp(x ** 2 + 2 * y ** 2)


def alpha(a):
    vector = np.dot(grad(x, y).transpose(), A0)
    x_a = x - vector[0][0] * a
    y_a = y - vector[0][1] * a
    return f(x_a, y_a)


def grad(x, y):

    diffX = -1 + 2 * np.exp(x ** 2 + 2 * y ** 2) * x + (2 * x) / np.sqrt(1 + 2 * x ** 2 + y ** 2)
    diffY = -1 + 4 * np.exp(x ** 2 + 2 * y ** 2) * y + y / np.sqrt(1 + 2 * x ** 2 + y ** 2)

    grad = np.array([[diffX, diffY]])
    return grad.transpose()


# Input data

x = -0.5
y = 0.5
h = 2
a0 = 0.1
eps = 10**(-5)
A0 = np.array([[1, 0], [0, 1]])
xk = np.array([x, y])
Ak = A0
k = 0
print("Results")
while True:
    print("Iteration", k)
    print("Point x", xk)
    print("Matrix\n", Ak)
    x = xk[0]
    y = xk[1]
    A0 = Ak
    gradientX = grad(x, y)
    gradientXk = grad(xk[0], xk[1])
    norma = np.sqrt(gradientXk[0] ** 2 + gradientXk[1] ** 2)
    if norma <= eps:
        print(xk)
        break
    if norma > eps:
        min = half_divide(alpha, DSK_algorithm(alpha, a0, h), eps)
        tempVector = np.dot(gradientXk.transpose(), A0)
        xk[0] = x - tempVector[0][0] * min
        xk[1] = y - tempVector[0][1] * min
        gradientXk = grad(xk[0], xk[1])
        gk = gradientXk - gradientX
        delta_xk = np.array([[xk[0] - x, xk[1] - y]])
        dxk__dxk_transope = np.dot(delta_xk.transpose(), delta_xk)
        dxk_transope_gk = np.dot(delta_xk, gk)
        gk__gk_transpose = np.dot(gk, gk.transpose())
        Ak_gk__gk_transpose = np.dot(A0, gk__gk_transpose)
        Ak_gk__gk_transpose_Ak = np.dot(Ak_gk__gk_transpose, A0)
        gk_transpose_Ak = np.dot(gk.transpose(), A0)
        gk_transpose_Ak_gk = np.dot(gk_transpose_Ak, gk)
        Ak = A0 + (dxk__dxk_transope / dxk_transope_gk) - (Ak_gk__gk_transpose_Ak / gk_transpose_Ak_gk)


    k = k + 1

    if not np.sqrt((xk[0] - x) ** 2 + (xk[1] - y) ** 2) > eps:
        break
