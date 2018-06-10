import numpy as np
import pywt
from matplotlib import image as mpimg
from matplotlib import pyplot as plt


def ista_algorithm(H, y, l):
    # Calculating the alpha term from the dictionary H
    alpha = np.max(np.linalg.eig(H.T.dot(H))[0])
    x = np.zeros(y.shape)
    # Setting the error as maximum for the first iteration
    j_old = np.finfo(np.float64).max

    while True:
        # Applying soft threshold
        x = pywt.threshold(x + ((H.T.dot(y - H.dot(x))) / alpha), l / (2 * alpha), mode='soft')
        # Applying cost function to minimize x
        j = np.power(np.linalg.norm(y - H.dot(x), 2), 2) + (l * np.linalg.norm(x, 1))
        if (j_old - j) / j_old < 1e-7:
            break
        j_old = j
    return x


def DCT_basis_gen(N):
    # Creating a matrix of alpha values of shape NxN
    a = np.vstack((np.full((1, N), fill_value=np.sqrt(1 / N)), np.full((N - 1, N), fill_value=np.sqrt(2 / N))))
    # Creating a matrix of cosine terms of shape NxN
    kn = np.outer(np.arange(0, N), 2 * (np.arange(0, N)) + 1)
    cos_mtx = np.cos(kn * np.full((N, N), fill_value=(np.pi / (2 * N))))
    return a * cos_mtx


def run_1d_ista():
    # Create a basis vector for N=16
    H = DCT_basis_gen(16)

    # A randomly generated sparse vector
    x = np.zeros(16)
    x[np.random.choice(16, 5)] = 1

    # Generating a noise vector
    w = np.random.normal(0, np.var(x) / 10, 16)

    # Adding noise to the sparse vector
    y = H.dot(x) + w

    # Applying ISTA on noisy sparse vector for different lambda values
    ista_result = {}
    lambda_range = np.arange(0, 2.4, 0.4)
    for lamb in lambda_range:
        ista_result[lamb] = ista_algorithm(H, y, lamb)

    # Plotting the results for different lambda values
    length_of_subplots = len(lambda_range) + 2
    plt.figure(figsize=(10, 30))
    for i in range(length_of_subplots):
        if i == 0:
            plt.subplot(length_of_subplots, 1, i + 1)
            plt.stem(x)
            plt.grid()
            plt.title('Original Signal')
        elif i == 1:
            plt.subplot(length_of_subplots, 1, i + 1)
            plt.stem(y)
            plt.grid()
            plt.title('Noisy Signal')
        else:
            plt.subplot(length_of_subplots, 1, i + 1)
            plt.stem(ista_result[lambda_range[i - 2]])
            plt.grid()
            plt.title('Lambda = {:.1f}'.format(lambda_range[i - 2]))
    plt.tight_layout()
    plt.savefig('1d_ista_result.png', bbox_inches='tight')
    plt.show()


def inv_transform(coeffs, wavelet):
    return pywt.waverec2(coeffs, wavelet)


def fwd_transform(data, wavelet):
    return pywt.wavedec2(data, wavelet, level=3)


def ista_algorithm_2d(wavelet, noisy_image, l):
    # Create a matrix of 0s and getting it's wavelet coefficients
    x = fwd_transform(np.zeros(noisy_image.shape), wavelet)

    # Converting coefficients to array for calculation
    x_arr, x_slices = pywt.coeffs_to_array(x)

    # Creating a list of errors
    error_log = []
    # Setting inital error to maximum
    error_old = np.finfo(np.float64).max
    # Run loop until error condition is satisfied
    while True:
        # Soft thresholding
        x_arr = pywt.threshold(x_arr + pywt.coeffs_to_array(
            fwd_transform(noisy_image - inv_transform(x, wavelet), wavelet))[0], l / 2, mode='soft')

        # Converting array back to coefficients
        x = pywt.array_to_coeffs(x_arr, x_slices, 'wavedec2')

        # Calculating error
        error = np.power(np.linalg.norm(noisy_image - inv_transform(x, wavelet), 2), 2) + np.linalg.norm(x_arr, 1) * l
        error_log.append(error)
        # Error condition for the loop to break
        if (error_old - error) / error_old < 1e-7:
            break
        error_old = error

    return inv_transform(x, wavelet), error_log


def run_2d_ista():
    # Loading input image
    x = mpimg.imread('lena512gray.png')[:, :, 0] * 255

    # Setting a seed to fix random noise
    np.random.seed(11)
    y = x + np.random.normal(0, np.var(x) / 100, x.shape)

    # Obtaining wavelet transforms of noisy image
    y_haar = pywt.coeffs_to_array(fwd_transform(y, 'haar'))[0]
    y_db8 = pywt.coeffs_to_array(fwd_transform(y, 'db8'))[0]

    # Plotting the images for a fixed noise value
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.title('Noisy Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(y_haar, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Haar Wavelet transform of noisy image')
    plt.subplot(2, 2, 4)
    plt.imshow(y_db8, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('DB8 Wavelet transform of noisy image')
    plt.tight_layout()
    plt.savefig('initial_plot.png', bbox_inches='tight')
    plt.show()

    # finding and plotting denoised image for different lambda values and wavelets
    for wavelet in ['haar', 'db8']:
        for lam in np.linspace(50, 100, 3):
            x_denoised = ista_algorithm_2d(wavelet, y, lam)
            plt.figure()
            plt.suptitle('Lambda = {} - Wavelet = {}'.format(lam, wavelet))
            plt.subplot(1, 2, 1)
            plt.imshow(pywt.coeffs_to_array(fwd_transform(x_denoised, wavelet))[0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Wavelet transform of denoised image')
            plt.subplot(1, 2, 2)
            plt.imshow(x_denoised, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Denoised image')
            plt.tight_layout()
            plt.savefig('2d_ista_result_wavelet_{}_lambda_{}.png'.format(wavelet, lam), bbox_inches='tight')
            plt.show()

    # Obtaining error vs iteration number graph with lambda = 75
    _, error_log_haar = ista_algorithm_2d('haar', y, 75)
    _, error_log_db8 = ista_algorithm_2d('db8', y, 75)
    x_axis_haar = np.arange(0, len(error_log_haar)) + 1
    x_axis_db8 = np.arange(0, len(error_log_db8)) + 1
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.xticks(x_axis_haar)
    plt.stem(x_axis_haar, error_log_haar)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Haar Wavelet with lambda = 75')
    plt.subplot(1, 2, 2)
    plt.xticks(x_axis_db8)
    plt.stem(x_axis_db8, error_log_db8)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('DB8 Wavelet with lambda = 75')
    plt.tight_layout()
    plt.savefig('error_log.png', bbox_inches='tight')
    plt.show()

# Running ISTA functions
run_1d_ista()
run_2d_ista()
