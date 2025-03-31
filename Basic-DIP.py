import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reading & Displaying Images
def read_and_display_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# Color Space Conversion
def convert_color_spaces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return gray, hsv

# Filtering & Smoothing
def apply_filters(image):
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    median_blur = cv2.medianBlur(image, 5)
    return gaussian_blur, median_blur

# Geometric Transformations
def geometric_transformations(image):
    resized = cv2.resize(image, (300, 300))
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(image, 1)
    return resized, rotated, flipped

# Edge Detection
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return canny, sobelx, laplacian

# Histogram Equalization
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

# Drawing on Images
def draw_on_image(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle((50, 50, 200, 200), outline="red", width=5)
    draw.text((60, 60), "Hello!", fill="blue")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Morphological Operations
def morphological_operations(image):
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return eroded, dilated

# Main Function
if __name__ == "__main__":
    # Corrected file path with raw string
    image_path = r"C:\Users\Admin\Desktop\Git demo\Images\sample.jpg"  # Replace with your image path
    image = read_and_display_image(image_path)

    if image is None:
        print("Exiting: Unable to load the image.")
    else:
        # Color Space Conversion
        gray, hsv = convert_color_spaces(image)

        # Filtering & Smoothing
        gaussian_blur, median_blur = apply_filters(image)

        # Geometric Transformations
        resized, rotated, flipped = geometric_transformations(image)

        # Edge Detection
        canny, sobelx, laplacian = edge_detection(image)

        # Histogram Equalization
        equalized = histogram_equalization(image)

        # Drawing on Images
        drawn_image = draw_on_image(image)

        # Morphological Operations
        eroded, dilated = morphological_operations(image)

        # Save processed images to the output folder
        cv2.imwrite("output/gray.jpg", gray)
        cv2.imwrite("output/hsv.jpg", hsv)
        cv2.imwrite("output/gaussian_blur.jpg", gaussian_blur)
        cv2.imwrite("output/median_blur.jpg", median_blur)
        cv2.imwrite("output/resized.jpg", resized)
        cv2.imwrite("output/rotated.jpg", rotated)
        cv2.imwrite("output/flipped.jpg", flipped)
        cv2.imwrite("output/canny.jpg", canny)
        cv2.imwrite("output/sobelx.jpg", sobelx)
        cv2.imwrite("output/laplacian.jpg", laplacian)
        cv2.imwrite("output/equalized.jpg", equalized)
        cv2.imwrite("output/drawn_image.jpg", drawn_image)
        cv2.imwrite("output/eroded.jpg", eroded)
        cv2.imwrite("output/dilated.jpg", dilated)

        print("Processing complete! Check the 'output' folder for results.")