import cv2
import matplotlib.pyplot as plt

# Step 1: Download the images
# Assuming the images are in the working directory
image_path_1 = '5th_July_map-view.png'
image_path_2 = '6th_July_map-view.png'

# Step 2: Load the images as grayscale images
img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded properly
if img1 is None or img2 is None:
    print('Could not open or find the images')
    exit(0)

# Step 3: Create a SIFT feature detector and detect keypoints in both images
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# Draw keypoints
img1_with_keypoints = cv2.drawKeypoints(img1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_with_keypoints = cv2.drawKeypoints(img2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot the images
plt.subplot(121), plt.imshow(img1_with_keypoints, cmap='gray'), plt.title('5th_July_map-view')
plt.subplot(122), plt.imshow(img2_with_keypoints, cmap='gray'), plt.title('6th_July_map-view')
plt.show()
