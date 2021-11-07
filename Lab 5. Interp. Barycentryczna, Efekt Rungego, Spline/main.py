import numpy as np
import scipy
import pickle

#from typing import Union, List, Tuple


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów

    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    if type(x) is not np.ndarray or type(y) is not np.ndarray:
        return None
    if np.shape(x) != np.shape(y):
        return None
    a = np.zeros(len(x)-1)
    b = np.zeros(len(x)-1)
    for i in range(0,len(x)-1):
        a[i]=(y[i+1]-y[i])/(x[i+1]-x[i])
        b[i]=y[i]-a[i]*x[i]
    return a,b



def cubic_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline trzeciego stopnia.
    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów
    Results:
    (a0,a1,a2,a3) - krotka zawierająca współczynniki funkcji wielomianowych"""

    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return None
    if len(x.shape) != 1 or len(y.shape) != 1:
        return None
    if x.shape != y.shape:
        return None
    if len(x.shape) != 1:
        return None

    h = np.zeros(len(y))
    d = np.zeros(len(y))
    lam = np.zeros(len(y))
    ro = np.zeros(len(y))
    m = np.zeros(len(y))
    a3 = np.zeros(len(y) - 1)
    a2 = np.zeros(len(y) - 1)
    a1 = np.zeros(len(y) - 1)
    a0 = np.zeros(len(y) - 1)
    b3 = np.zeros(len(y) - 1)
    b2 = np.zeros(len(y) - 1)
    b1 = np.zeros(len(y) - 1)
    b0 = np.zeros(len(y) - 1)

    for i in range(len(y) - 1):
        h[i] = x[i + 1] - x[i]
        d[i] = (y[i + 1] - y[i]) / h[i]

    for i in range(len(y) - 1):
        lam[i + 1] = h[i + 1] / (h[i] + h[i + 1])
        ro[i + 1] = h[i] / (h[i] + h[i + 1])

    for i in range(len(y) - 2):
        m[i + 1] = 3 * (d[i + 1] - d[i]) / (h[i + 1] + h[i]) - m[i] * ro[i + 1] / 2 - m[i + 2] * lam[i + 1] / 2

    for i in range(len(y) - 1):
        b3[i] = (m[i + 1] - m[i]) / 6 * h[i]
        b2[i] = m[i] / 2
        b1[i] = d[i] - (h[i] * (2 * m[i] + m[i + 1])) / 6
        b0[i] = y[i]

    for i in range(len(y) - 1):
        a3[i] = b3[i]
        a2[i] = b2[i] - 3 * b3[i] * x[i]
        a1[i] = b1[i] - 2 * b2[i] * x[i] + 3 * b3[i] * x[i] * x[i]
        a0[i] = b0[i] - b1[i] * x[i] + b2[i] * x[i] * x[i] - b3[i] * x[i] * x[i] * x[i]
    return (a0, a1, a2, a3)