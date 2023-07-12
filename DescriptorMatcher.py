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

# Step 4: Create a DescriptorMatcher and compute correspondences between the features of both images
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

# Match the descriptors
matches = matcher.match(descriptors_1, descriptors_2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches
match_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot the matched images
plt.imshow(match_img)
plt.title('Matches')
plt.show()
