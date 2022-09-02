import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def main():
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    n = 30
    L = 10
    h = L/n
    x1 = np.array([i*h-0.5*L for i in range(n)])
    x2 = np.array([i*h-0.5*L for i in range(n)])
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x = np.stack([x1_grid.flatten(), x2_grid.flatten()], axis=1)
    x_shape = x1_grid.shape

    y = multivariate_normal.pdf(x=x, mean=mean, cov=cov)

    input = np.stack([x1_grid.flatten(), x2_grid.flatten()], axis=1)
    output = y.flatten()

    with open('input.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(input)

    with open('target.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(output)

    plt.figure()
    plt.contour(x1_grid, x2_grid, y.reshape(x_shape))
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()