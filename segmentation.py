


import cv2
import numpy as np

def segment(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the intensity range for the color #e0e0e0 (gray value 224)
    lower_e0e0e0 = 220
    upper_e0e0e0 = 228

    # Define the intensity range for the color #fafafa (gray value 250)
    lower_fafafa = 245
    upper_fafafa = 255

    # Create masks for the two intensity ranges
    mask_e0e0e0 = cv2.inRange(gray, lower_e0e0e0, upper_e0e0e0)
    mask_fafafa = cv2.inRange(gray, lower_fafafa, upper_fafafa)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_e0e0e0, mask_fafafa)

    # Apply morphological operations to smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to further smooth the mask
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return result
