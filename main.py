import cv2
import numpy as np


def draw_perpendicular_line_from_centroid(output, centroid, angle, contour):
    theta = np.deg2rad(angle)
    ray_length = 1000
    p1 = (
        int(centroid[0] + ray_length * np.cos(theta)),
        int(centroid[1] + ray_length * np.sin(theta)),
    )
    p2 = (
        int(centroid[0] - ray_length * np.cos(theta)),
        int(centroid[1] - ray_length * np.sin(theta)),
    )
    cv2.line(output, p1, p2, (255, 0, 255), 2)


def detect_and_color_objects_and_grab_points(
    image_path, resize=True, area_threshold_ratio=0.1
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    if resize:
        resize_factor = 700 / min(image.shape[:2])
        image = cv2.resize(
            image,
            (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor)),
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        int(max(0, (1.0 - 0.5) * np.median(gray))),
        int(min(255, (1.0 + 0.5) * np.median(gray))),
    )
    combined = cv2.addWeighted(
        gray,
        0.5,
        cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1000),
        0.2,
        0,
    )
    contours, _ = cv2.findContours(
        cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = image.copy()

    for contour in [
        cnt
        for cnt in contours
        if cv2.contourArea(cnt) > area_threshold_ratio * image.shape[0] * image.shape[1]
    ]:
        cv2.drawContours(
            output, [contour], -1, np.random.randint(0, 255, 3).tolist(), 2
        )
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(output, centroid, 5, (0, 0, 255), -1)

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                draw_perpendicular_line_from_centroid(
                    output, centroid, ellipse[2], contour
                )

    cv2.imwrite("objects_with_centroid_and_lines.png", output)
    cv2.imshow("Detected Objects", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "0002.jpg"
detect_and_color_objects_and_grab_points(image_path)
