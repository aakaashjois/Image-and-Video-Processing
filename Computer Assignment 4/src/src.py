from cv2 import imread
from cv2.xfeatures2d import SIFT_create

from utils import match_image_points, create_matched_pairs, plot_image, circle_interest_points, transform_image, \
    modify_image, draw_keypoints, make_panorama, get_harris_interest_points, get_sift_descriptor

# Read test image
im = imread('.\sample image\lena512gray.png', 0)

# Rotate and scale the test image
rotate_image_1, rotate_image_2, scale_image_1, scale_image_2 = modify_image(im)

# Identify the Harris interest points for the test image
interest_points = get_harris_interest_points(1, im, 50)

# Plot the identified interest points on the image
circled_image = circle_interest_points(interest_points, im)
plot_image(circled_image, 'Harris Interest Points')

# Identify the Harris interest points for the left rotated test image
rotate_interest_points_1 = get_harris_interest_points(1, rotate_image_1, 50)

# Plot the identified interest points on the image
circle_rotate_image_1 = circle_interest_points(rotate_interest_points_1, rotate_image_1)
plot_image(circle_rotate_image_1, 'Harris Interest Points - Rotate 2 degrees')

# Identify the Harris interest points for the right rotated test image
rotate_interest_points_2 = get_harris_interest_points(1, rotate_image_2, 50)

# Plot the identified interest points on the image
circle_rotate_image_2 = circle_interest_points(rotate_interest_points_2, rotate_image_2)
plot_image(circle_rotate_image_2, 'Harris Interest Points - Rotate -2 degrees')

# Identify the Harris interest points for the scaled up test image
scale_interest_points_1 = get_harris_interest_points(1, scale_image_1, 50)

# Plot the identified interest points on the image
circle_scale_image_1 = circle_interest_points(scale_interest_points_1, scale_image_1)
plot_image(circle_scale_image_1, 'Harris Interest Points - Scale up by 2')

# Identify the Harris interest points for the scaled down test image
scale_interest_points_2 = get_harris_interest_points(1, scale_image_2, 50)

# Plot the identified interest points on the image
circle_scale_image_2 = circle_interest_points(scale_interest_points_2, scale_image_2)
plot_image(circle_scale_image_2, 'Harris Interest Points - Scale down by 2')

# Get SIFT descriptors for the rotated images
sift_1 = get_sift_descriptor(rotate_image_1, rotate_interest_points_1)
sift_2 = get_sift_descriptor(rotate_image_2, rotate_interest_points_2)

# Get the matched pairs of points
matched_pairs = create_matched_pairs(sift_1, sift_2, 0.5)

# Plot the matched pairs of feature points
large_image = match_image_points(rotate_image_1, rotate_image_2, rotate_interest_points_1, rotate_interest_points_2,
                                 matched_pairs)
plot_image(large_image, 'Matched Interest Points')

# Read left and right image for panoramic stitching
im_left = imread('.\sample image\DT_left.jpg')
im_right = imread('.\sample image\DT_right.jpg')

# Create the SIFT feature point detector object
sift = SIFT_create()

# Identify the keypoints and SIFT descriptors
skp_left = sift.detect(im_left)
skp_right = sift.detect(im_right)
skp_left, sd_left = sift.compute(im_left, skp_left)
skp_right, sd_right = sift.compute(im_right, skp_right)

# Plot the keypoints on the image
keypoints_left = draw_keypoints(im_left, skp_left)
plot_image(keypoints_left, 'Keypoints on Left Image')
keypoints_right = draw_keypoints(im_right, skp_right)
plot_image(keypoints_right, 'Keypoints on Right Image')

# Adjust the descriptors to be of equal sizes
if sd_left.size < sd_right.size:
    sd_right = sd_right[0:sd_left.shape[0], 0:sd_left.shape[1]]
elif sd_left.size >= sd_right.size:
    sd_left = sd_left[0:sd_right.shape[0], 0:sd_right.shape[1]]

# Identify the matched pairs of points
match_points = create_matched_pairs(sd_left, sd_right, 0.5)

# Plot the matched pairs on the image
large_image = match_image_points(im_left, im_right, skp_left, skp_right, match_points, True)
plot_image(large_image, 'Matched SIFT Points')

# Transform the right image to be the same plane as left image
transformed = transform_image(im_left, im_right, skp_left, skp_right, match_points)
plot_image(transformed, 'Transformed Image')

# Stitch the left and right image to create the panorama
panorama = make_panorama(transformed, im_left)
plot_image(panorama, 'Stitched Image')
