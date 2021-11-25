import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from sklearn.mixture import GaussianMixture
def run_test_percentile(img):

    start_time = time.time()
    for i in range(10):
        # img = cv2.resize(img, (300,300))
        # i_min, i_max = np.percentile(img, [1, 99])
        a= np.median(img)
    end_time = time.time()
    average_time = (end_time - start_time)/10
    print('running time :' + str(average_time))

    return

def gmm_hist(x):
    # hist = plt.hist(x, bins=256)
    # n, bins = hist[0], hist[1]
    # Fit GMM
    # gmm = GaussianMixture(n_components=5)
    # gmm = gmm.fit(X=np.expand_dims(x, 1))

    # Evaluate GMM
    # gmm_x = bins
    # gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))

    md = np.median(x)
    md1 = np.mean(x[x>md])

    # Plot histograms and gaussian curves
    fig, ax = plt.subplots()
    # ax.hist(x, 256, [0, 255], normed=True)
    ax.hist(x, bins=256, density=True)
    plt.axvline(x=md,c='r')
    plt.axvline(x=md1,c='r')

    # ax.plot(gmm_x, gmm_y, color="crimson", lw=2, label="GMM")

    ax.set_ylabel("Frequency")
    ax.set_xlabel("Pixel Intensity")

    plt.legend()

    plt.show()
    return