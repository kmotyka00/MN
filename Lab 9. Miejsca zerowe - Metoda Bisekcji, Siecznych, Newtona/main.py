import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if type(a) not in [int, float] or type(b) not in [int, float]:
        print("ab")
        return None
    if type(epsilon) is not float or epsilon<0:
        print("eps", type(epsilon))
        return None
    if type(iteration) is not int or iteration<=0:
        print("itera")
        return None
    if not isfunction(f):
        print("isfunc")
        return None
    if f(a)*f(b) >=0:
        print("signs")
        return None
    n = 1
    while n<iteration:
        c = (a+b)/2
        if f(c) == 0 or np.abs((b-a))/2 < epsilon:
            return c, n
        n = n + 1
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    
    print("finish")
    return (a+b)/2, n

#print(bisection(0.5,1,fun,0.001,100))
#Ustalam parametry, które będę później podawał do funkcji
a = -1
c = 0.5
b = 1
epsilon = 0.01
max_iter = 100

#Znajduję miejsca zerowe za pomocą
print(fun(a))
print(fun(c))
#METODY BISEKCJI
sol1_x1 = bisection(a,c,fun,epsilon,max_iter)
print (sol1_x1)

def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if type(a) not in [int, float] or type(b) not in [int, float]:
        return None
    if type(epsilon) is not float or epsilon<=0:
        return None
    if type(iteration) is not int or iteration<=0:
        return None
    if not isfunction(f):
        return None
    if f(a)*f(b) >= 0:
        return None
    a_n = a
    b_n = b
    n = 1
    for i in range(1,iteration+1):
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n <= epsilon:
            return m_n, n
        else:
            return None
        n=n+1
    return a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n)), n-1
     

#secant(0.5,1,fun,0.001,1000)

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if type(a) not in [int, float] or type(b) not in [int, float]:
        return None
    if type(epsilon) is not float or epsilon<=0:
        return None
    if type(iteration) is not int or iteration<=0:
        return None
    if not isfunction(f) or not isfunction(df) or not isfunction(ddf):
        return None
    if f(a)*f(b) >=0:
        return None
    if a>b:
        return None
    if np.sign(df(b)) != np.sign(df(a)) or np.sign(ddf(b)) != np.sign(ddf(a)):
        return None
    x0 = (a+b)/2
    xn = x0
    for n in range(0,iteration):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            return xn, n
        dfxn = df(xn)
        if dfxn == 0:
            return None
        xn = xn - fxn/dfxn
        
    return None
print(newton(fun,dfun,ddfun,0.5,1,0.001,1000))
