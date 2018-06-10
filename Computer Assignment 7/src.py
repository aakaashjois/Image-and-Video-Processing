import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def perform_intracoding(frame, step):
    """
    Perform intracoding on the frame based on the number of steps given.
    """
    BLOCK_SIZE = 8
    l, m = np.shape(frame)
    result = np.zeros((l, m))
    for i in range(1, l - BLOCK_SIZE, BLOCK_SIZE):
        for j in range(1, m - BLOCK_SIZE, BLOCK_SIZE):
            block = frame[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            vector = np.concatenate((np.reshape(frame[i - 1, j - 1:j + BLOCK_SIZE], (1, -1)),
                                     np.reshape(frame[i:i + BLOCK_SIZE, j - 1], (1, -1))),
                                    axis=1)
            var, error, prediction = find_best_intra_prediction(vector, block)
            dct_var = cv.dct(np.array(error, dtype='float32'))
            result[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = prediction + cv.idct(perform_quantization(dct_var, step))
    return result


def matcher(image, template):
    """
    Matches image with the provided template.
    """
    l = np.shape(template)
    p = np.shape(image)
    match = np.zeros((2,))
    error = 1e10

    for i in range(0, p[0] - l[0]):
        for j in range(0, p[1] - l[1]):
            temp1 = np.reshape(image[i:i + l[0], j:j + l[1]] - template, [l[0] * l[1], 1])
            temp = np.sum(np.square(temp1))
            if temp < error:
                match[0] = i
                match[1] = j
                error = temp
            match = match.astype(np.int)
    return image[match[0]:match[0] + l[0], match[1]:match[1] + l[1]]


def perform_intra_prediction(mode, pre_values, block):
    """
    Performs intra predictions based on the mode.
    """
    if mode == 0:
        mat = np.matrix(pre_values[:, 1:9])
        for i in range(1, 8):
            mat = np.append(mat, np.matrix(pre_values[:, 1:9]), axis=0)
        error_block = block - mat
        var = np.var(error_block)
        return var, error_block, mat
    elif mode == 1:
        mat = np.matrix(pre_values[:, 9:17])
        for i in range(1, 8):
            mat = np.append(mat, np.matrix(pre_values[:, 9:17]), axis=0)
        mat = mat.T
        error_block = block - mat
        var = np.var(error_block)
        return var, error_block, mat
    elif mode == 2:
        mat = np.mean(pre_values) * np.ones((8, 8))
        error_block = block - mat
        var = np.var(error_block)
        return var, error_block, mat


def find_best_intra_prediction(pre_values, block):
    """
    Finds and returns the best intra prediction values.
    """
    var0, err0, A0 = perform_intra_prediction(0, pre_values, block)
    var1, err1, A1 = perform_intra_prediction(1, pre_values, block)
    var2, err2, A2 = perform_intra_prediction(2, pre_values, block)
    ind = np.argmin([var0, var1, var2])
    if ind == 0:
        return var0, err0, A0
    elif ind == 1:
        return var1, err1, A1
    elif ind == 2:
        return var2, err2, A2


def perform_quantization(dct_coef, step_size):
    """
    Quantize the image based on the DCT Coefficients and step size.
    """
    return step_size * np.round(dct_coef / step_size)


def calculate_psnr(I, K):
    """
    Finds the PSNR of the two images.
    """
    m, n = np.shape(I)
    MSE = np.sum(np.square(I - K)) / (m * n)
    return 20 * np.log10(np.max(I)) - 10 * np.log10(MSE)


if __name__ == '__main__':
    frame1 = cv.imread('foreman010.jpg', 0)
    frame2 = cv.imread('foreman020.jpg', 0)

    frame1 = cv.resize(frame1, (560, 560))
    frame2 = cv.resize(frame2, (560, 560))

    l, m = np.shape(frame1)
    avg_psnr = []
    avg_zero = []

    for step in tqdm(range(1, 17)):
        f1_hat = perform_intracoding(frame1, step)
        psnr = []
        final_zero = []
        num_zeros = 0
        psnr.append(calculate_psnr(frame1, f1_hat))

        f3_hat = np.zeros((l, m))
        temp1 = cv.copyMakeBorder(f1_hat, 20, 20, 20, 20, cv.BORDER_REPLICATE)
        final_error = np.zeros((l, m))
        final_predict = np.zeros((l, m))
        for i in range(1, l - 8, 8):
            for j in range(1, m - 8, 8):
                block = frame2[i:i + 8, j:j + 8]
                vector = np.concatenate(
                    (np.reshape(frame2[i - 1, j - 1:j + 8], (1, -1)), np.reshape(frame2[i:i + 8, j - 1], (1, -1))),
                    axis=1)
                var_intra, err_intra, predict_intra = find_best_intra_prediction(vector, block)
                matched = matcher(temp1[i:i + 40, j:j + 40], block)
                error_match = block - matched
                var_match = np.var(error_match)

                if var_match < var_intra:
                    final_error[i:i + 8, j:j + 8] = error_match
                    final_predict[i:i + 8, j:j + 8] = matched
                else:
                    final_error[i:i + 8, j:j + 8] = err_intra
                    final_predict[i:i + 8, j:j + 8] = predict_intra

        for i in range(0, l - 8, 8):
            for j in range(0, m - 8, 8):
                dct_var = cv.dct(np.array(final_error[i:i + 8, j:j + 8], dtype='float32'))
                out = perform_quantization(dct_var, step)
                idct_var = cv.idct(out)
                f3_hat[i:i + 8, j:j + 8] = final_predict[i:i + 8, j:j + 8] + idct_var
                idx = np.argwhere(dct_var != 0)
                num_zeros = num_zeros + idx.shape[1]

        final_zero.append(num_zeros)

        f2_hat = np.zeros((l, m))
        temp2 = cv.copyMakeBorder(f3_hat, 20, 20, 20, 20, cv.BORDER_REPLICATE)

        for i in range(1, l - 8, 8):
            for j in range(1, m - 8, 8):
                block = frame1[i:i + 8, j:j + 8]
                vector = np.concatenate(
                    (np.reshape(frame1[i - 1, j - 1:j + 8], (1, -1)), np.reshape(frame1[i:i + 8, j - 1], (1, -1))),
                    axis=1)
                var_intra, err_intra, predict_intra = find_best_intra_prediction(vector, block)

                matched1 = matcher(temp1[i:i + 40, j:j + 40], block)
                error_match1 = block - matched1
                var_match1 = np.var(error_match1)

                matched3 = matcher(temp2[i:i + 40, j:j + 40], block)
                error_match3 = block - matched3
                var_match3 = np.var(error_match3)

                idx = np.argmin([var_intra, var_match1, var_match3])

                if idx == 0:
                    final_error[i:i + 8, j:j + 8] = err_intra
                    final_predict[i:i + 8, j:j + 8] = predict_intra
                elif idx == 1:
                    final_error[i:i + 8, j:j + 8] = error_match1
                    final_predict[i:i + 8, j:j + 8] = matched1
                else:
                    final_error[i:i + 8, j:j + 8] = error_match3
                    final_predict[i:i + 8, j:j + 8] = matched3

        for i in range(0, l - 8, 8):
            for j in range(0, m - 8, 8):
                dct_var = cv.dct(np.array(final_error[i:i + 8, j:j + 8], dtype='float32'))
                out = perform_quantization(dct_var, step)
                idct_var = cv.idct(out)
                f3_hat[i:i + 8, j:j + 8] = final_predict[i:i + 8, j:j + 8] + idct_var
                idx = np.argwhere(dct_var != 0)
                num_zeros = num_zeros + idx.shape[1]

        final_zero.append(num_zeros)
        psnr.append(calculate_psnr(frame1, f2_hat))
        psnr.append(calculate_psnr(frame2, f3_hat))

        avg_psnr.append(np.mean(np.array(psnr)))
        avg_zero.append(np.mean(np.array(final_zero)))

    plt.figure()
    plt.imshow(f3_hat, cmap='gray')
    plt.figure()
    plt.plot(np.arange(1, 17), avg_psnr)
    plt.show()
