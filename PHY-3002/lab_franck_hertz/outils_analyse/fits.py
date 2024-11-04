import sigfig as sig
from scipy.optimize import curve_fit
import numpy as np
import scipy.stats as sci_stats


def linear_regression(x: np.array, y: np.array) -> tuple:
    """
    Cette méthode effectue une régression linéaire des y en fonction des x.

    :param x: numpy array 1d associé aux valeurs indépendantes
    :param y: numpy array 1d associé aux valeurs dépendantes

    :return: un tuple de 5 éléments ayant la pente, l'ordonnée à L'origine, le R (coefficient de corrélation,
             l'incertitude sur la pente et l'incertitude sur l'ordonnée à l'origine
    """
    result = sci_stats.linregress(x, y)
    (slope, intercept, r_value, p_value, slope_std) = result
    intercept_std = result.intercept_stderr
    rounded_slope_std = round_any(slope_std, sigfigs=1)
    rounded_intercept_std = round_any(intercept_std, sigfigs=1)

    return slope, intercept, r_value, rounded_slope_std, rounded_intercept_std


def round_any(a: np.array or np.ndarray or float, **kwargs) -> float:
    """
    Fonction permettant d'arrondir un seul nombre ou un array entier

    :param a: la ou les valeurs à arrondir
    :param kwargs: Le kwargs sigfigs permet de choisir le nombre de chiffres significatifs

    :return: Le ou les nombres arrondis
    """
    if a is not float:
        return round_array(a, **kwargs)
    return round_np_float(a, **kwargs)


def round_array(a: np.array or np.ndarray, **kwargs):
    """
    Fonction permetant d'arrondir un array entier

    :param a: Les valeurs à arrondir
    :param kwargs: Le kwargs sigfigs permet de choisir le nombre de chiffres significatifs

    :return: Les nombres arrondis
    """
    return np.vectorize(round_np_float)(a, **kwargs)


def round_np_float(value: np.float_, **kwargs) -> float:
    """
    Fonction permettant d'arrondir un seul nombre

    :param value: La valeur à arrondir
    :param kwargs: Le kwargs sigfigs permet de choisir le nombre de chiffres significatifs

    :return: Le nombre arrondi
    """
    return sig.round(float(value), **kwargs)


def gaus(x, a, x0, sigma):
    """
    Fonction (modèle) de gaussienne classique.

    :param x: Valeurs indépendantes
    :param a: Amplitude de la gaussienne
    :param x0: Moyenne de la gaussienne
    :param sigma: écart-type de la gaussienne

    :return: Les valeurs dépendantes de la gaussienne
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gaussian_fit(x: np.array, y: np.array, a_estimation, mu_estimation, sigma_estimation):
    """
    Cette fonction applique un fit gaussien sur les valeurs x et y. Attention, cette méthode est
    très sensible aux valeurs estimées de l'amplitude, de la moyenne et de l'écart-type de la gaussienne.
    Cette méthode retourne les paramètres optimaux du fit gaussien. Pour n'avoir qu'un seul pic à la fois,
    appliquer cette fonction sur des régions de l'array complet (en utilisant du slicing [:]).

    :param x: Valeurs indépendantes (attention d'utiliser des valeurs qui ne contiennent qu'un seul pic)
    :param y: Valeurs dépendantes (attention d'utiliser des valeurs qui ne contiennent qu'un seul pic)
    :param a_estimation: Estimation de l'amplitude (devrait être l'amplitude associé à l'estimation préalable du pic valeurs[estimation du pic])
    :param mu_estimation: Estimation de la moyenne (devrait être l'index associé à l'estimation préalable du pic)
    :param sigma_estimation: Estimation de l'écart-type (vous pouvez souvent considérer une variance de 1)

    :return: Un tuple dont le premier élément est un array de trois éléments [amplitude optimale, moyenne optimale,
             ecart-type optimal] tandis que le deuxième un un array de trois éléments ayant les incertitudes associées.
    """
    popt, pcov = curve_fit(gaus, x, y, [a_estimation, mu_estimation, sigma_estimation])
    err = np.sqrt(np.diag(pcov))

    return popt, err
