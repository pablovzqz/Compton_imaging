import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/scratch/pablovzqz/MELT/src/')

from core import create_interpolator



def plot_light_table():

    xs, ys, interp  = create_interpolator()
    xn = np.arange(xs.min(), xs.max() + 1e-9, 15.0)
    yn = np.arange(ys.min(), ys.max() + 1e-9, 15.0)
    XN, YN = np.meshgrid(xn, yn, indexing='ij')
    Zn = interp(np.column_stack([XN.ravel(), YN.ravel()]))

    Zn_2d = Zn.reshape(XN.shape)
    plt.pcolormesh(xn, yn, Zn_2d.T, shading='nearest', cmap='inferno')
    plt.colorbar(label='Normalized Factor')
    plt.gca().set_aspect('equal')

def plot_deconvolved(ml_em, y, centers_x, centers_y, path_bin_x, path_bin_y):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap_g = plt.get_cmap("gnuplot2").copy()
    cmap_g.set_under("white")

    mask = ml_em[0].ravel() > 0.05 * ml_em[0].ravel().max()   
    order = np.argsort(ml_em[0].ravel()[mask])

    sc = axes[0].scatter(
        np.asarray(np.load(path_bin_x))[mask][order],
        np.asarray(np.load(path_bin_y))[mask][order],
        c=ml_em[0].ravel()[mask][order],
        s=12,
        cmap=cmap_g,
        linewidths=0,
        vmin=0.1*ml_em[0].ravel().max(),
        vmax=ml_em[0].ravel().max(),)


    axes[0].set_title("ML-EM reconstruction")
    axes[0].set_xlabel("X [mm]")
    axes[0].set_ylabel("Y [mm]")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-500, 500)
    axes[0].set_ylim(-500, 500)

    fig.colorbar(sc, ax=axes[0], label="Reconstructed charge")

    cx = np.asarray(centers_x)   # 66 valores
    cy = np.asarray(centers_y)   # 66 valores

    # Malla completa de centros, mismo orden que y.reshape(66, 66)
    XX, YY = np.meshgrid(cx, cy, indexing="ij")
    XX = XX.ravel()
    YY = YY.ravel()

    y_flat = np.asarray(y).ravel()

    mask_y = y_flat > 0.01 * y_flat.max()   # ajusta el umbral
    order_y = np.argsort(y_flat[mask_y])

    sc1 = axes[1].scatter(
        XX[mask_y][order_y],
        YY[mask_y][order_y],
        c=y_flat[mask_y][order_y],
        s=12,
        cmap=cmap_g,
        linewidths=0,
        vmin=0.1*y_flat.max(),
        vmax=y_flat.max(),
    )
    fig.colorbar(sc1, ax=axes[1], label="Measured charge")

    axes[1].set_title("Measured SiPM signal")
    axes[1].set_xlabel("X [mm]")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-500, 500)
    axes[1].set_ylim(-500, 500)