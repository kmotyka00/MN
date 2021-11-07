import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(x) is not np.ndarray:
        return None
    
    return P.polyfromroots(x)

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(a) is not np.ndarray:
        return None

    for i in range(0,len(a)):
        a[i] = a[i] + 10**(-10)*np.random.random_sample()
    x = P.polyroots(a)
    return a, x


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(wsp) is not np.ndarray:
        return None
    n = len(wsp)
    frob_matrix = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if i==j-1:
                frob_matrix[i][j] = 1
    frob_matrix[n-1] = -wsp
    eigenvalues= np.linalg.eigvals(frob_matrix)
    schur = scipy.linalg.schur(frob_matrix)

    wsp_all = np.concatenate((np.ndarray(1),wsp))
    polyroot_result = P.polyroots(wsp_all)

    return frob_matrix, eigenvalues, schur, polyroot_result


