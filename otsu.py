import cv2

# Load the image
image_path = "0001.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's binarization
ret, otsu_thresholded_image = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Save the result
otsu_image_path = "otsu_binary_0001.jpg"
cv2.imwrite(otsu_image_path, otsu_thresholded_image)

# Optionally, display the image (requires a GUI environment)
cv2.imshow("Otsu Binarized Image", otsu_thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
