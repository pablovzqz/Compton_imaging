import tables as tb
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator


def get_hits(path: str, chunk_size: int = 1_000_000) -> pd.DataFrame:

    dfs = []

    with tb.open_file(path, mode="r") as f:

        table = f.root.hits.table

        for start in range(0, table.nrows, chunk_size):

            stop = min(start + chunk_size, table.nrows)

            data = table.read(start, stop)

            dfs.append(pd.DataFrame.from_records(data))

    df = pd.concat(dfs, ignore_index=True)

    return df

def get_light_table(mode: str) -> pd.DataFrame:

    if mode == "PMTs":
        path = '/data/pablovzqz/NEXT100-data/compton_imaging/data/NEXT100_MC_v07_11_00_May2026_S2_LT.h5'

    elif mode == "SiPMs":
        path = '/data/pablovzqz/NEXT100-data/compton_imaging/data/NEXT100_MC_v07_11_00_May2026_SiPM_LT.h5'

    else:
        raise ValueError("Mode must be 'PMTs' or 'SiPMs'")

    with tb.open_file(path, 'r') as h5file:
        data = h5file.root.LT.LightTable[:]
        lt = pd.DataFrame(data)

    return lt

def create_interpolator():

        lt = get_light_table('PMTs')
        light_map = pd.DataFrame({'x': lt['x'], 'y': lt['y'], 'factor': lt['PmtR11410_total']})

        xs = np.sort(light_map['x'].unique())
        ys = np.sort(light_map['y'].unique())

        Z = (light_map.pivot(index='x', columns='y', values='factor')
                    .reindex(index=xs, columns=ys)
                    .values)

        interp = RegularGridInterpolator((xs, ys), Z, method='linear', bounds_error=False, fill_value=None)

        return xs, ys, interp

def create_interpolated_light_table(pitch: float = 15.0) -> RegularGridInterpolator:

    xs, ys, interp = create_interpolator()

    xn = np.arange(xs.min(), xs.max() + 1e-9, pitch)
    yn = np.arange(ys.min(), ys.max() + 1e-9, pitch)
    XN, YN = np.meshgrid(xn, yn, indexing='ij')
    Zn = interp(np.column_stack([XN.ravel(), YN.ravel()]))

    df = pd.DataFrame({'x': XN.ravel(), 'y': YN.ravel(), 'factor': Zn})

    return pd.DataFrame({'x': df['x'], 'y': df['y'], 'factor': df['factor']/df.factor.sum()})


def create_CKDTree() -> cKDTree:

    light_map = create_interpolated_light_table(pitch=15.0)
    lt_voxel_centers = np.column_stack([light_map.x.to_numpy(), light_map.y.to_numpy()])
    tree = cKDTree(lt_voxel_centers)
 
    return tree

def find_path_files(idx: int) -> tuple:

    path_sensors            = [f'/data/pablovzqz/NEXT100-data/compton_imaging/light_tables/sensor_response_by_bin_sigma_{i}.npy' for i in np.arange(0, 1400, 100)]
    path_bins_x             = [f'/data/pablovzqz/NEXT100-data/compton_imaging/data/bin_x_sigma_{i}.npy'                  for i in np.arange(0, 1400, 100)]
    path_bins_y             = [f'/data/pablovzqz/NEXT100-data/compton_imaging/data/bin_y_sigma_{i}.npy'                  for i in np.arange(0, 1400, 100)]
    path_bin_ids            = [f'/data/pablovzqz/NEXT100-data/compton_imaging/data/bin_ids_sigma_{i}.npy'                for i in np.arange(0, 1400, 100)]
    path_n_electrons_by_bin = [f'/data/pablovzqz/NEXT100-data/compton_imaging/data/n_electrons_by_bin_sigma_{i}.npy'     for i in np.arange(0, 1400, 100)]

    return path_sensors[idx], path_bins_x[idx], path_bins_y[idx], path_bin_ids[idx], path_n_electrons_by_bin[idx]
    