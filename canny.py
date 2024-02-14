import cv2

# Load the image
image_path = "0001.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and smoothen the image
# (5, 5) is the kernel size for the Gaussian Blur. The numbers should be odd.
# The larger the kernel size, the more the blur.
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply the Canny filter
# 50 and 150 are the minimum and maximum threshold values
# These thresholds can be adjusted to detect stronger or weaker edges
canny_edges = cv2.Canny(blurred_image, 50, 80)

# Save the result
cv2.imwrite("canny_edges_0002.jpg", canny_edges)

# Optionally, display the image (requires a GUI environment)
cv2.imshow("Canny Edges", canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
