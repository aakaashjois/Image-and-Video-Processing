from cv2 import warpPerspective, getRotationMatrix2D, circle, line, resize, findHomography, warpAffine, drawKeypoints, \
    RANSAC, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
from matplotlib import pyplot as plt
from numpy import pad, sqrt, square, multiply, rad2deg, arctan2, where, argmax, roll, array, ones, full, argpartition, \
    vstack, unravel_index, arange, floor, exp, outer, sum, concatenate, min as npmin, argmin, max as npmax, unique, \
    hsplit, count_nonzero
from numpy.linalg import norm
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


def get_sift_descriptor(image, interest_points):
    # Taking a patch of 16x16 pixels
    patch_width = 16

    # Sigma is taken as half the patch size
    patch_sigma = patch_width / 2

    # Get the Gaussian filter of same size as the patch
    patch_gaussian_filter = generate_gaussian_filter(patch_sigma, patch_width)

    # Create orientation bins of size 8
    orientations = arange(0, 8)

    # Zero pad the image
    image_padded = pad(image, patch_width, 'constant', constant_values=0)

    sift_descriptor = []

    for x, y in interest_points:

        # Offset the index based on the zero padding
        x = patch_width + x
        y = patch_width + y

        # Obtain the patch and generate the HoG
        patch = image_padded[x - 8: x + 8, y - 8: y + 8]
        patch_x_gradient, patch_y_gradient = get_image_gradients(patch_sigma, patch_width, patch)
        patch_magnitude = sqrt(square(patch_x_gradient) + square(patch_y_gradient))
        patch_weighted = multiply(patch_magnitude, patch_gaussian_filter)
        patch_orientation = (rad2deg(arctan2(patch_y_gradient, patch_x_gradient)) + 360) % 360
        patch_quantized = get_binned(patch_orientation)
        hist_of_grad = [patch_weighted[where(patch_quantized == orientation)].sum() for orientation in orientations]

        # Find the dominant orientation for the patch
        dominant_orientation = orientations[argmax(hist_of_grad)]

        # Create subpatch of size 4x4 pixels
        subpatch_width = 4

        # Sigma is taken as half the patch size
        subpatch_sigma = subpatch_width / 2

        # Get the Gaussian filter of same size as the patch
        subpatch_gaussian_filter = generate_gaussian_filter(subpatch_sigma, subpatch_width)

        histogram_of_gradients = []

        for i in range(0, 4):
            for j in range(0, 4):
                # Get the patch and generate the HoG
                subpatch = patch[4 * i: 4 * i + 4, 4 * j:4 * j + 4]
                subpatch_x_gradient, subpatch_y_gradient = get_image_gradients(subpatch_sigma, subpatch_width, subpatch)
                subpatch_magnitude = sqrt(square(subpatch_x_gradient) + square(subpatch_y_gradient))
                subpatch_weighted = multiply(subpatch_magnitude, subpatch_gaussian_filter)
                subpatch_orientation = (rad2deg(arctan2(subpatch_y_gradient, subpatch_x_gradient)) + 360) % 360
                subpatch_quantized = get_binned(subpatch_orientation)
                hist_of_grad = [subpatch_weighted[where(subpatch_quantized == orientation)].sum() for orientation in
                                orientations]

                # Roll the HoG such that dominant orientation is at the first index
                histogram_of_gradients.append(roll(hist_of_grad, dominant_orientation))

        # Normalize and threshold the HoG
        histogram_of_gradients = array(histogram_of_gradients).flatten() / norm(histogram_of_gradients)
        histogram_of_gradients[where(histogram_of_gradients > 0.2)] = 0.2
        histogram_of_gradients = histogram_of_gradients / norm(histogram_of_gradients)

        sift_descriptor.append(histogram_of_gradients)

    return array(sift_descriptor)


def get_harris_interest_points(sigma, image, n):
    # Obtain the gradient images
    image_x_gradient, image_y_gradient = get_image_gradients(sigma, 1 + (4 * sigma), image)

    image_gradient_x_square = image_x_gradient * image_x_gradient
    image_gradient_y_square = image_y_gradient * image_y_gradient
    image_gradient_x_times_y = image_x_gradient * image_y_gradient

    # Obtain Gaussian filter
    gaussian_filter = generate_gaussian_filter(2 * sigma, 1 + (6 * sigma), return_vector=False)

    # Apply gaussian filter by convolution
    image_gradient_x_square_smooth = conv2d(image_gradient_x_square, gaussian_filter)
    image_gradient_y_square_smooth = conv2d(image_gradient_y_square, gaussian_filter)
    image_gradient_x_times_y_smooth = conv2d(image_gradient_x_times_y, gaussian_filter)

    # Create a matrix of ones of size equal to Gaussian filter
    ones_array = ones(gaussian_filter.shape)

    # Apply convolution with ones to get summation for 9 values around each vector
    moment_x_square = conv2d(image_gradient_x_square_smooth, ones_array)
    moment_y_square = conv2d(image_gradient_y_square_smooth, ones_array)
    moment_x_times_y = conv2d(image_gradient_x_times_y_smooth, ones_array)

    # Generate the determinant and trace of the moment components
    moment_determinant = (moment_x_square * moment_y_square) - (moment_x_times_y * moment_x_times_y)
    moment_trace = moment_x_square + moment_y_square

    # Obtain moment matrix
    moment = moment_determinant - 0.06 * square(moment_trace)

    # Zero padding to find local maxima in neighborhood of 9 elements
    moment_padded = pad(moment, 1, 'constant', constant_values=0)
    interest_points = full(moment_padded.shape, fill_value=False)
    for u in range(1, moment_padded.shape[0] - 1):
        for v in range(1, moment_padded.shape[1] - 1):
            neighborhood = moment_padded[u - 1: u + 2, v - 1: v + 2]
            if npmax(neighborhood) == moment_padded[u, v]:
                interest_points[u, v] = True
    interest_points = interest_points[1: -1, 1: -1]

    # Picking n largest features
    moment_interest_points = moment * interest_points
    interest_points_index = argpartition(moment_interest_points.flatten(), -n)[-n:]
    indices = vstack(unravel_index(interest_points_index, moment_interest_points.shape)).T
    return indices


def conv2d(image, filt):
    # Perform 2D Convolution with symmetric boundary condition
    return convolve2d(image, filt, mode='same', boundary='symm')


def generate_gaussian_filter(sigma, length, return_vector=False):
    # Check the length of the window to generate a vector
    z = arange(-floor(length / 2), floor(length / 2) + (0 if length % 2 == 0 else 1))

    # Generate 1D Gaussian vector
    gaussian_1d = exp((-square(z)) / (2 * square(sigma)))

    # Take outer product of 1D Gaussian vector to create a 2D Gaussian matrix
    gaussian_2d = outer(gaussian_1d, gaussian_1d)

    # Normalize the 2D Gaussian
    gaussian_2d = gaussian_2d / sum(gaussian_2d)

    if return_vector:
        return gaussian_2d, z
    else:
        return gaussian_2d


def generate_gaussian_gradient_filter(sigma, length):
    # Obtain the 2D Gaussian filter
    gaussian, z = generate_gaussian_filter(sigma, length, return_vector=True)

    # Multiply with the scalar values after taking derivative
    # gaussian = (-1 / np.square(sigma)) * gaussian
    gaussian = - multiply(gaussian, square(sigma))
    # Multiply with vector depending on direction
    grad_x = multiply(z, gaussian)
    grad_y = multiply(z, gaussian.T).T

    return grad_x, grad_y


def get_image_gradients(sigma, length, image):
    # Obtain the 2D Gaussian derivative matrix
    x_gradient_filter, y_gradient_filter = generate_gaussian_gradient_filter(sigma, length)

    # Get gradient images by convolution
    image_x_gradient = conv2d(image, x_gradient_filter)
    image_y_gradient = conv2d(image, y_gradient_filter)

    return image_x_gradient, image_y_gradient


def get_binned(image):
    # Set number of bins
    n = 8

    # Calculate quantization value
    quantize_value = 360 / n

    # Bin the images at each pixel
    image_binned = floor((image + quantize_value / 2) / quantize_value)

    # Set bin = n values to bin = 0 because it is circular
    image_binned[image_binned == n] = 0

    return image_binned


def circle_interest_points(interest_points, image):
    # Fix the coordinates to work with OpenCV
    interest_points[:, [0, 1]] = interest_points[:, [1, 0]]

    # Add circles on the original image
    for circle_ind in interest_points:
        circle(image, tuple(circle_ind), 7, (255, 255, 255), 2)

    return image


def modify_image(image):
    # Find the center of the image
    center = tuple(array(image.shape[::-1]) // 2)

    # Rotate the image by 45 and -45 degrees
    rotate_image_1 = warpAffine(image, getRotationMatrix2D(center, 2, 1), image.shape[::-1])
    rotate_image_2 = warpAffine(image, getRotationMatrix2D(center, -2, 1), image.shape[::-1])

    # Scale image down and up by a factor of 2
    scale_image_1 = resize(image, tuple(array(image.shape[::-1]) // 2))
    scale_image_2 = resize(image, tuple(array(image.shape[::-1]) * 2))

    return rotate_image_1, rotate_image_2, scale_image_1, scale_image_2


def plot_image(image, title):
    # Plot image in a new figure
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('.\output image\\' + title, bbox_inches='tight')
    plt.show()


def match_image_points(image_1, image_2, feature_points_1, feature_points_2, matched_pairs, inbuilt=False):
    # Match the feature points between two images
    global feature_point_1, feature_point_2
    large_image = concatenate((image_1, image_2), axis=1)

    if inbuilt:
        for pair in matched_pairs:
            left_index = feature_points_1[pair[0]].pt
            right_index = feature_points_2[pair[1]].pt
            feature_point_1 = (int(left_index[0]), int(left_index[1]))
            feature_point_2 = (int(right_index[0] + image_1.shape[1]), int(right_index[1]))
    else:
        feature_points_2 = feature_points_2 + array([0, image_1.shape[1]])
        for matched_pair in matched_pairs:
            feature_point_1 = tuple(feature_points_1[matched_pair[0]][::-1])
            feature_point_2 = tuple(feature_points_2[matched_pair[1]][::-1])

    line(large_image, feature_point_1, feature_point_2, (255, 255, 255), 2)
    return large_image


def create_matched_pairs(sift_features_1, sift_features_2, r):
    # Match SIFT Feature Descriptors and return the indices of the matched pairs
    distances = cdist(sift_features_1, sift_features_2)
    min_dist_1 = npmin(distances, axis=1)
    min_dist_1_ind = argmin(distances, axis=1)
    distances2 = distances.flatten()
    max_value_index = (distances.shape[0] * arange(distances.shape[0])) + min_dist_1_ind
    distances2[max_value_index] = npmax(distances) + 1
    distances2 = distances2.reshape(distances.shape)
    min_dist_2 = npmin(distances2, axis=1)
    ratios = (min_dist_1 / min_dist_2) < r
    accepted_points_index = min_dist_1_ind * ratios
    match_point_2, match_point_1 = unique(accepted_points_index, return_index=True)
    return list(zip(match_point_1, match_point_2))


def transform_image(im_left, im_right, skp_left, skp_right, matched_pairs):
    # Transform the image based on planar homography
    src_points = []
    dst_points = []

    for x, y in matched_pairs:
        src_points.append((int(skp_right[y].pt[0]), int(skp_right[y].pt[1])))
        dst_points.append((int(skp_left[x].pt[0]), int(skp_left[x].pt[1])))

    (H, status) = findHomography(array(src_points), array(dst_points), RANSAC)
    return warpPerspective(im_right, H, (im_left.shape[1] + im_right.shape[1], im_left.shape[0]))


def make_panorama(transformed, image):
    # Stitch the panoramic image using the original left image and transformed right image
    transformed[0: image.shape[0], 0: image.shape[1]] = image

    # Remove the black extension on the side of the image
    border = transformed.shape[1]
    for i, c in enumerate(hsplit(transformed, transformed.shape[1])):
        if count_nonzero(c) == 0:
            border = i - 1
            break
    return transformed[:, 0: border]


def draw_keypoints(image, keypoints):
    return drawKeypoints(image, keypoints, None, flags=DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
