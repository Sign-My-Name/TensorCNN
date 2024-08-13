import mediapipe as mp
import cv2
import numpy as np



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def augment_image(image: np.array, to_cut: bool, to_transform: bool, to_rotate: bool, size: tuple = None) -> np.array:
    """Applies the required transformations to the image."""
    if to_rotate:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if to_cut:
        image = cut(image=image, size=size)
    if to_transform:
        image = transform(image)
    return image
def draw_hand_skeleton(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image
    return image


def isolate_and_crop_hand(image, padding=60):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        #image = draw_hand_skeleton(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for hand_landmarks in results.multi_hand_landmarks:
            points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                      for landmark in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)

            # Apply padding via dilation
            kernel = np.ones((padding * 2, padding * 2), np.uint8)
            padded_mask = cv2.dilate(mask, kernel, iterations=1)

            # Create a black background image
            black_background = np.zeros_like(image)

            # Isolate the hand by combining it with the black background
            isolated_hand = cv2.bitwise_and(image, image, mask=padded_mask)
            final_image = cv2.bitwise_or(black_background, isolated_hand)

            # Calculate bounding box for the isolated hand with padding
            x, y, w, h = cv2.boundingRect(padded_mask)
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x_start + w + 2 * padding, image.shape[1])
            y_end = min(y_start + h + 2 * padding, image.shape[0])

            # Crop the image to the bounding box with padding
            cropped_image = final_image[y_start:y_end, x_start:x_end]
            return cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return None


def cut(image, size=None):
    """
    :param image: cv2 image
    :param size: (width,height)
    :return: cv2 image
    """
    image = image.astype(np.uint8)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = hands.process(rgb_image)
    if processed_image.multi_hand_landmarks:
        hand_landmarks = processed_image.multi_hand_landmarks[0]
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        val_to_adjust = max(image.shape[0], image.shape[1]) * 0.08
        x_min_adjust = int(x_min * image.shape[1] - val_to_adjust)
        y_min_adjust = int(y_min * image.shape[0] - val_to_adjust)
        x_max_adjust = int(x_max * image.shape[1] + val_to_adjust)
        y_max_adjust = int(y_max * image.shape[0] + val_to_adjust)
        if x_min_adjust < 0:
            x_min_adjust = int(x_min * image.shape[1] - 15)
            if x_min_adjust < 0:
                x_min_adjust = 0
        if y_min_adjust < 0:
            y_min_adjust = int(y_min * image.shape[0] - 15)
            if y_min_adjust < 0:
                y_min_adjust = 0
        x_min = x_min_adjust
        y_min = y_min_adjust
        x_max = x_max_adjust
        y_max = y_max_adjust
        hand_region = rgb_image[y_min:y_max, x_min:x_max]
        hand_region_uint8 = hand_region.astype(np.uint8)
        if size is not None:
            hand_region_uint8 = cv2.resize(hand_region_uint8, dsize=size)
        return hand_region_uint8
    else:
        return None


def transform(image):
    if image is None or image.shape[0] == 0:
        return None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Gray scale

    sharpening_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    sharpened_image = cv2.filter2D(image_gray, -1, sharpening_kernel)  # applying sharpening kernal
    return sharpened_image