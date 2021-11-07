import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
import math

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if type(v) not in [int, float, list, np.ndarray] or type(v_aprox) not in [int, float, list, np.ndarray]: #ważne żeby było list a nie List bo to konkretny rodzaj z List
        return np.nan
    if type(v) is int and type(v_aprox) is list:
        return [abs(v - x) for x in v_aprox]
    if type(v) is int and type(v_aprox) is list:
        return [abs(v_aprox - x) for x in v]
    if type(v) is list and type(v_aprox) is list:
        if len(v) == len(v_aprox):
            return [abs(v[i] - v_aprox[i]) for i in range(len(v))]
        else: 
            return np.nan
    if type(v) is np.ndarray and type(v_aprox) is np.ndarray and v.shape[0] != v_aprox.shape[0]:
            return np.nan
    return abs(v - v_aprox) #float i float, int i float, int i int, ndarray i ndarray

    
def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if type(v) not in [int, float, list, np.ndarray] or type(v_aprox) not in [int, float, list, np.ndarray]: #ważne żeby było list a nie List bo to konkretny rodzaj z List
        return np.nan
    if (type(v) is np.ndarray or type(v) is list):
        if 0 in v:
            return np.nan
    elif v == 0:
        return np.nan
    if type(v) is int and type(v_aprox) is list:
        return [abs((v - x)/v) for x in v_aprox]
    if type(v) is int and type(v_aprox) is list:
        return [abs((v_aprox - x)/x) for x in v]
    if type(v) is list and type(v_aprox) is list:
        if len(v) == len(v_aprox):
            return [abs((v[i] - v_aprox[i]) / v[i]) for i in range(len(v))]
        else: 
            return np.nan
    if type(v) is np.ndarray and type(v_aprox) is np.ndarray and v.shape[0] != v_aprox.shape[0]:
            return np.nan
    return abs(np.divide(v - v_aprox,v)) #float i float, int i float, int i int, ndarray i ndarray


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if type(n) is not int or type(c) not in [int, float]:
        return np.nan

    b = 2**n
    P1 = (b-b)+c
    P2 = (b+c)-b
    return absolut_error(P1,P2)


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if type(n) is not int or type(x) not in [int, float] or n<=0:
        return np.nan
    
    res:float = 0
    for i in range(0,n):
        res=res + 1/np.math.factorial(i)*x**i

    return res


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if type (k) is not int or type(x) not in [int, float]:
        return np.nan
    if k < 0:
        return np.nan
    if x == 0:
        return np.cos(0)
    if k == 1 or k == 0:
        return np.cos(k*x)
    else:
        return (2*np.cos(x)*coskx1(k-1,x)) - coskx1(k-2,x) 

# if type(k) is not int or (type(x) is not int and type(x) is not float):
#         return np.nan
#     if k < 0:
#         return np.nan
#     if x == 0:
#         return np.cos(0)
#     if k == 1 or k == 0:
#         return np.cos(k*x)
#     return (2 * coskx1(1,x) * coskx1(k-1,x)) - (coskx1(k-2,x) * x)

def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    if type(k) is not int or type(x) not in [int, float]:
        return np.nan
    if k < 0:
        return np.nan
    if x == 0:
        return np.cos(0),np.sin(0)
    if k == 1 or k == 0:
        return np.cos(k*x), np.sin(k*x)
    return np.cos(x) * coskx2(k-1,x)[0] - np.sin(x) * coskx2(k-1,x)[1], np.sin(x) * coskx2(k-1,x)[0] + np.cos(x) * coskx2(k-1,x)[1]


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if type(n) is not int or n<=0:
        return np.nan
    pi_approx = 0
    for i in range(1,n+1):
        pi_approx = pi_approx + 1/i**2
    return np.sqrt(6*pi_approx)



# a=np.array([[1, 1, 4, 5],
#  [5, 1, 2, 3],
#  [7, 1, 2, 3],
#  [2, 2, 2, 2],
#  [1, 2, 7, 1]])

# b=np.array([1,2,3,4])
# print(a)
# print(b)
# print(len(a))
# print(np.size(a))
# print(np.shape(a))
# print(np.size(b))

# print(p_diff(0,3))
# print(np.math.factorial(5))
# x = 4.128714922026218
# n =12 
# print(exponential(x,12))



# res_exact = np.exp(4.51231145)
# n1=exponential(4.51231145,1)
