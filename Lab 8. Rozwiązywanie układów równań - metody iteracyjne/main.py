import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    
    #SPRAWDZENIE WARUNKÓW ZAD
    if type(m) is not int or m<=0:
        return None
    # MACIERZ WARTOŚCI RANDOMOWYCH
    A = np.random.randint(0,100,size=(m,m))
    b = np.random.randint(0,9,m)
    # WEKTOR WARTOŚCI Z DIAGONALI
    v_diag = np.array(np.diag(A))
    #USUWANIE WARTOSCI Z DIAGONALI
    A_temp = A - np.eye(m)*v_diag
    #SPRWADZANIE CZY WART NA DIAG WIEKSZE
    for i in range(m):
        #WIERSZ
        if np.abs(v_diag[i]) < np.sum(A_temp[i,:]):
            v_diag[i] = v_diag[i] + np.sum(A_temp[i,:]) - np.abs(v_diag[i]) + np.random.randint(0,100)
        #KOLUMNA
        if np.abs(v_diag[i]) < np.sum(A_temp[:,i]):
            v_diag[i] = v_diag[i] + np.sum(A_temp[:,i]) - np.abs(v_diag[i]) + np.random.randint(0,100)
    for i in range(m):
        A[i,i] = v_diag[i]

    return A, b


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(A) is not np.ndarray or len(np.shape(A)) != 2:
        return None
    if np.shape(A)[0] != np.shape(A)[1]:
        return None
    m = np.shape(A)[0]
    v_diag = np.array(np.diag(A))
    A_temp = A - np.eye(m)*v_diag
    for i in range(m):
        #WIERSZ
        if np.abs(v_diag[i]) < np.sum(A_temp[i,:]):
            return False
        #KOLUMNA
        if np.abs(v_diag[i]) < np.sum(A_temp[:,i]):
            return False
    return True
        


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) is not int or m<=0:
        return None 

    A = np.random.randint(0,100,size=(m,m))
    b = np.random.randint(0,9,m)
    return (A+np.transpose(A)), b


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(A) is not np.ndarray or len(np.shape(A)) != 2:
        return None
    if np.shape(A)[0] != np.shape(A)[1]:
        return None
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if A[i][j]!=A[j][i]:
                return False
    return True


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if type(A) is not np.ndarray or type(b) is not np.ndarray or type(x_init) is not np.ndarray:
        return None
    if type(epsilon) not in [None, float] or type(maxiter) not in [None, int]:
        return None
    if len(np.shape(A)) != 2:
        return None
    if np.shape(A)[0] != np.shape(A)[1] or np.shape(A)[0] != len(b):
        return None
    if len(x_init) != len(b) or epsilon < 0 or maxiter<0:
        return None
    D = np.diag(np.diag(A)) #dwa razy żeby była macierz a nie wektor
    LU = A - D
    x = x_init
    D_inv = np.diag(1/np.diag(D))
    resid = []
    n = 0
    for i in range(maxiter):
        x_new = np.dot(D_inv,b - np.dot(LU,x))
        r_norm = np.linalg.norm(x_new - x)
        resid.append(r_norm)
        n = i+1
        if r_norm<epsilon:
            return x_new, n
        x = x_new
    return x, n

