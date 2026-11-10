import numpy as np
from tqdm import tqdm

def ML_EM(guess, y, sensor_response_by_bin, cluster, iterations = 100):

    sensor_response_by_bin = np.maximum(sensor_response_by_bin, 1e-12)
    y = np.maximum(y, 1e-12)

    ll = np.zeros(iterations)

    for i in tqdm(range(iterations)):
        
        scale = guess @ sensor_response_by_bin

        guess = guess * 1 /np.sum(sensor_response_by_bin, axis = 1) * ((y /scale) @ sensor_response_by_bin.T)

        mu = sensor_response_by_bin.T @ guess
        if i % 5 == 0:
            log_likelihood = np.sum(y * np.log(mu) - mu)
            ll[i] = -2 *log_likelihood

    return guess/np.sum(guess) * cluster.E.sum(), ll

