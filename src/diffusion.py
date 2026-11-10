import numpy as np


def D_from_D_star(D_L_star_um_sqrtcm, T, T0, Z_P, Z0_P, P_bar, v_d_mm_us):

    '''
       Transforma el coeficiente de difusión reducido al coeficiente de difusión.
    
        Args:
            D_L_star_um_sqrtcm (float): D*_L en µm/sqrt(cm).
            T (float): Temperatura.
            T0 (float): Temperatura de referencia.
            Z_P (float): Coeficiente de compreisibilidad.
            Z0_P (float): Coeficiente de compreisibilidad de referencia.
            P_bar (float): Presión en bar.
            v_d_mm_us (float): Velocidad de deriva en mm/µs.

        Returns:
            float: Coeficiente de difusión D_L en cm^2/s.
        '''
    D_star = D_L_star_um_sqrtcm * 1e-4      # µm/sqrt(cm) -> sqrt(cm)
    v_d = v_d_mm_us * 1e5                   # mm/µs -> cm/s

    return D_star**2 * (T / T0) * (Z_P / Z0_P) * v_d / (2.0 * P_bar) * 4

def sigma(D, t):
    """
    Calcula la desviación estándar de la distribución gaussiana de la difusión.

    Args:
        D (float): coeficiente de difusión
        t (float): tiempo máximo de deriva en el detector.

    returns:
        sigma (float): desviación estándar de la distribución gaussiana
    """
    return np.sqrt(2.0 * D * (t*1e-6))*10 