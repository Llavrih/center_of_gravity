import cv2

# Load the image
image_path = "0001.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and smoothen the image
# (5, 5) is the kernel size for the Gaussian Blur. The numbers should be odd.
# The larger the kernel size, the more the blur.
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 3)

# Apply binary thresholding on the blurred image
# Here, 127 is the threshold value, and 255 is the maximum value
# cv2.THRESH_BINARY is the type of thresholding
(thresh, black_and_white_image) = cv2.threshold(
    blurred_image, 140, 255, cv2.THRESH_BINARY
)

# Save the black and white image
bw_image_path = "black_and_white_0002_blurred.jpg"
cv2.imwrite(bw_image_path, black_and_white_image)

# Optionally, display the image (requires a GUI environment)
cv2.imshow("Black and White Blurred Image", black_and_white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
