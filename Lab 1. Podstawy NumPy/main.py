import math
import numpy as np
import scipy

def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r>=0.0 and h>= 0.0:
        return 2*math.pi*r**2+2*math.pi*r*h
    else:
        return np.nan


def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """   
    if type(n)!=int:
        return None
    if n==0:
        return None
    if n==1:
        return [1]
    if n>=1:
        vect=np.ones((n,1))
        a,b = (1,1)
        for i in range (n):
            vect[i]=a
            a,b = b, a+b
        return np.transpose(vect)
    if n<0:
        return None

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M=np.array([[a, 1, -a],
                [0, 1, 1],
                [-a, a, 1]])
    
    
    
    Mt= np.transpose(M)

    Mdet = np.linalg.det(M)
    if Mdet==0:
        Minv=np.nan
    else:
        Minv = np.linalg.inv(M)


    return (Minv, Mt, Mdet)

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if type(m)!=int or type(n)!=int:
        return None

    
    if m>0 and n>0:
        M=np.ones((m,n))
        for i in range (m):
            for j in range (n):
                if i>j:
                    M[i,j]=i
                else:
                    M[i,j]=j
        return M
    else:
        return None

