import numpy as np
import cv2
from matplotlib import pyplot as plt


def conv2d(image, filter):
    m, n = filter.shape
    padded_image = np.pad(image, m - 1, 'constant', constant_values=0)
    y, x = padded_image.shape
    y = y - m + 1
    x = x - n + 1
    conv_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            conv_image[i][j] = np.sum(padded_image[i:i + m, j:j + n] * filter)
    return conv_image


def image_input():
    path = input('Please enter the path of the image: ')
    if not path:
        print('Empty path')
        return
    return path


def filter_input():
    filter_size = int(input('Enter filter size: '))
    if not filter_size:
        print('Filter size invalid')
        return
    else:
        print('Filter is of size {}x{}'.format(filter_size, filter_size))
    filter = [float(element) for element in input('Please enter {} filter elements: '.format(filter_size ** 2)).split()]
    filter = np.array(filter)
    if not filter.size == filter_size ** 2:
        print('Filter size mismatch')
        return
    filter = filter.reshape((filter_size, filter_size))
    print(filter)
    return filter


def create_noise_image(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    return np.add(image, noise)


def create_gaussian_filter(filter_length, sigma):
    x = np.arange(-filter_length // 2 + 1, filter_length // 2 + 1)
    gauss = np.exp((-x ** 2) / (2 * sigma ** 2))
    gauss2 = np.outer(gauss.T, gauss)
    return gauss2 / np.sum(gauss2)


def create_average_filter(filter_length):
    return np.ones((filter_length, filter_length)).reshape([filter_length, filter_length]) / filter_length ** 2


def get_image_mag_response(image):
    fft = np.fft.fftshift(np.fft.fft2(image))
    return 20 * np.log(np.abs(fft))


def get_filter_freq_response(filter):
    f = np.zeros((100, 100))
    f[: filter.shape[0], :filter.shape[1]] = filter
    fft = np.fft.fftshift(np.fft.fft2(f))
    return np.abs(fft) + 1


def main():
    image = cv2.imread(image_input(), cv2.IMREAD_GRAYSCALE)
    filter = filter_input()
    conv_image = conv2d(image, filter)
    norm_image = ((conv_image - np.min(conv_image)) / (np.max(conv_image) - np.min(conv_image)) * 255).astype('uint8')
    cv2.imwrite('filtered_image.jpg', norm_image)
    plt.suptitle('Convolution with Filter')
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(norm_image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Filtered Image')
    plt.show()

    im_mag = get_image_mag_response(image)
    conv_mag = get_image_mag_response(conv_image)
    filt_freq_res = get_filter_freq_response(filter)
    plt.suptitle('Convolution - Magnitude Response')
    plt.subplot(3, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im_mag, cmap='gray')
    plt.title('Original Image')
    plt.subplot(3, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(conv_mag, cmap='gray')
    plt.title('Filtered Image')
    plt.subplot(3, 1, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(filt_freq_res, cmap='gray')
    plt.title('Filter Frequency Response')
    plt.savefig('magnitude_response.png', bbox_inches='tight')
    plt.show()
    cv2.waitKey(0)

    image = ((image - np.min(image)) / (np.max(image) - np.min(image)))

    noise_img_1 = create_noise_image(image, 0.01)
    noise_img_2 = create_noise_image(image, 0.1)

    noise_img_1 = ((noise_img_1 - np.min(noise_img_1)) / (np.max(noise_img_1) - np.min(noise_img_1)) * 255).astype('uint8')
    noise_img_2 = ((noise_img_2 - np.min(noise_img_2)) / (np.max(noise_img_2) - np.min(noise_img_2)) * 255).astype('uint8')

    avg_filt_1 = create_average_filter(5)
    avg_filt_2 = create_average_filter(7)
    avg_filt_3 = create_average_filter(9)

    gauss_filt_1 = create_gaussian_filter(5, 5 / 5)
    gauss_filt_2 = create_gaussian_filter(7, 7 / 5)
    gauss_filt_3 = create_gaussian_filter(9, 9 / 5)

    img_1_avg_filt_1 = conv2d(noise_img_1, avg_filt_1)
    img_1_avg_filt_2 = conv2d(noise_img_1, avg_filt_2)
    img_1_avg_filt_3 = conv2d(noise_img_1, avg_filt_3)

    img_1_gauss_filt_1 = conv2d(noise_img_1, gauss_filt_1)
    img_1_gauss_filt_2 = conv2d(noise_img_1, gauss_filt_2)
    img_1_gauss_filt_3 = conv2d(noise_img_1, gauss_filt_3)

    img_2_avg_filt_1 = conv2d(noise_img_2, avg_filt_1)
    img_2_avg_filt_2 = conv2d(noise_img_2, avg_filt_2)
    img_2_avg_filt_3 = conv2d(noise_img_2, avg_filt_3)

    img_2_gauss_filt_1 = conv2d(noise_img_2, gauss_filt_1)
    img_2_gauss_filt_2 = conv2d(noise_img_2, gauss_filt_2)
    img_2_gauss_filt_3 = conv2d(noise_img_2, gauss_filt_3)

    plt.suptitle('Average Filtering on Noise Image 1')
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Noise Image 1')
    plt.imshow(noise_img_1, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 1')
    plt.imshow(img_1_avg_filt_1, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 2')
    plt.imshow(img_1_avg_filt_2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 3')
    plt.imshow(img_1_avg_filt_3, cmap='gray')
    plt.savefig('avg_filt_1.png', bbox_inches='tight')
    plt.show()

    plt.suptitle('Average Filtering on Noise Image 2')
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Noise Image 2')
    plt.imshow(noise_img_2, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 1')
    plt.imshow(img_2_avg_filt_1, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 2')
    plt.imshow(img_2_avg_filt_2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Average Filter 3')
    plt.imshow(img_2_avg_filt_3, cmap='gray')
    plt.savefig('avg_filt_2.png', bbox_inches='tight')
    plt.show()

    plt.suptitle('Gaussian Filtering on Noise Image 1')
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Noise Image 1')
    plt.imshow(noise_img_1, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 1')
    plt.imshow(img_1_gauss_filt_1, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 2')
    plt.imshow(img_1_gauss_filt_2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 3')
    plt.imshow(img_1_gauss_filt_3, cmap='gray')
    plt.savefig('gauss_filt_1.png', bbox_inches='tight')
    plt.show()

    plt.suptitle('Gaussian Filtering on Noise Image 2')
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Noise Image 2')
    plt.imshow(noise_img_2, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 1')
    plt.imshow(img_2_gauss_filt_1, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 2')
    plt.imshow(img_2_gauss_filt_2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Gaussian Filter 3')
    plt.imshow(img_2_gauss_filt_3, cmap='gray')
    plt.savefig('gauss_filt_2.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
