import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = (112, 112)


def apply_white_balance(image):
    image_float = image.astype(np.float32)

    b_avg = np.mean(image_float[:, :, 0])
    g_avg = np.mean(image_float[:, :, 1])
    r_avg = np.mean(image_float[:, :, 2])

    avg = (b_avg + g_avg + r_avg) / 3.0

    image_float[:, :, 0] *= avg / (b_avg + 1e-7)
    image_float[:, :, 1] *= avg / (g_avg + 1e-7)
    image_float[:, :, 2] *= avg / (r_avg + 1e-7)

    return np.clip(image_float, 0, 255).astype(np.uint8)


def create_fast_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask_dark = gray < 245
    mask_saturated = saturation > 25
    mask_not_white = value < 248

    mask = np.logical_or(mask_dark, mask_saturated)
    mask = np.logical_or(mask, mask_not_white)
    mask = mask.astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    edges = cv2.Canny(blurred, 35, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(mask, edges)
    mask = cv2.bitwise_or(mask, otsu_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def preprocess_image(image):
    image = apply_white_balance(image)
    resized = cv2.resize(image, (192, 192))

    mask = create_fast_mask(resized)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 80
    ]

    if len(valid_contours) == 0:
        cropped = cv2.resize(resized, IMAGE_SIZE)
        cropped_mask = cv2.resize(mask, IMAGE_SIZE)
        return cropped, cropped_mask

    x_min = 192
    y_min = 192
    x_max = 0
    y_max = 0

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)

        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    padding = 16

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(192, x_max + padding)
    y_max = min(192, y_max + padding)

    cropped = resized[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    if cropped.size == 0:
        cropped = resized
        cropped_mask = mask

    cropped = cv2.resize(cropped, IMAGE_SIZE)
    cropped_mask = cv2.resize(cropped_mask, IMAGE_SIZE)

    return cropped, cropped_mask


def extract_hog_features(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)


def normalize_histogram(hist):
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_channel_statistics(cropped, mask):
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)

    mask_bool = mask > 0

    if np.sum(mask_bool) < 80:
        mask_bool = np.ones(mask.shape, dtype=bool)

    spaces = [
        (cropped.astype(np.float32), [255.0, 255.0, 255.0]),
        (hsv.astype(np.float32), [180.0, 255.0, 255.0]),
        (lab.astype(np.float32), [255.0, 255.0, 255.0])
    ]

    stats = []

    for space, scales in spaces:
        for channel_index, scale in enumerate(scales):
            values = space[:, :, channel_index][mask_bool] / scale

            stats.extend([
                float(np.mean(values)),
                float(np.std(values)),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75))
            ])

    return np.array(stats, dtype=np.float32)


def extract_color_histogram(cropped, mask):
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)

    if np.sum(mask > 0) < 80:
        mask = None

    hsv_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hsv_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    hsv_v = cv2.calcHist([hsv], [2], mask, [32], [0, 256])

    hsv_h = normalize_histogram(hsv_h)
    hsv_s = normalize_histogram(hsv_s)
    hsv_v = normalize_histogram(hsv_v)

    lab_l = cv2.calcHist([lab], [0], mask, [16], [0, 256])
    lab_a = cv2.calcHist([lab], [1], mask, [16], [0, 256])
    lab_b = cv2.calcHist([lab], [2], mask, [16], [0, 256])

    lab_l = normalize_histogram(lab_l)
    lab_a = normalize_histogram(lab_a)
    lab_b = normalize_histogram(lab_b)

    h, w = hsv.shape[:2]

    regions = [
        hsv[:h // 2, :],
        hsv[h // 2:, :],
        hsv[:, :w // 2],
        hsv[:, w // 2:]
    ]

    spatial_features = []

    for region in regions:
        region_h = cv2.calcHist([region], [0], None, [12], [0, 180])
        region_s = cv2.calcHist([region], [1], None, [12], [0, 256])
        region_v = cv2.calcHist([region], [2], None, [12], [0, 256])

        region_h = normalize_histogram(region_h)
        region_s = normalize_histogram(region_s)
        region_v = normalize_histogram(region_v)

        spatial_features.extend(region_h)
        spatial_features.extend(region_s)
        spatial_features.extend(region_v)

    features = np.concatenate([
        hsv_h,
        hsv_s,
        hsv_v,
        lab_l,
        lab_a,
        lab_b,
        np.array(spatial_features, dtype=np.float32),
        extract_channel_statistics(cropped, mask)
    ])

    return features.astype(np.float32)


def extract_lbp_features(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    features = []

    for radius in [1, 2]:
        points = 8 * radius

        lbp = local_binary_pattern(
            gray,
            points,
            radius,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, points + 3),
            range=(0, points + 2)
        )

        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)
        features.append(hist)

    return np.concatenate(features).astype(np.float32)


def extract_shape_features(cropped, mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 40
    ]

    if len(valid_contours) == 0:
        return np.zeros(25, dtype=np.float32)

    largest = max(valid_contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    x, y, w, h = cv2.boundingRect(largest)

    image_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    bbox_area = w * h

    area_ratio = area / image_area
    bbox_ratio = bbox_area / image_area
    aspect_ratio = w / (h + 1e-7)
    extent = area / (bbox_area + 1e-7)
    perimeter_ratio = perimeter / (2 * (IMAGE_SIZE[0] + IMAGE_SIZE[1]))
    width_ratio = w / IMAGE_SIZE[0]
    height_ratio = h / IMAGE_SIZE[1]
    center_x_ratio = (x + (w / 2.0)) / IMAGE_SIZE[0]
    center_y_ratio = (y + (h / 2.0)) / IMAGE_SIZE[1]

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-7)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    mask_bool = mask > 0

    if np.sum(mask_bool) < 40:
        mask_bool = np.ones(mask.shape, dtype=bool)

    mean_gray = np.mean(gray[mask_bool]) / 255.0
    mean_saturation = np.mean(hsv[:, :, 1][mask_bool]) / 255.0
    mean_value = np.mean(hsv[:, :, 2][mask_bool]) / 255.0

    circularity = (4.0 * np.pi * area) / ((perimeter * perimeter) + 1e-7)
    equivalent_diameter_ratio = np.sqrt((4.0 * area) / (np.pi + 1e-7)) / IMAGE_SIZE[0]

    edges = cv2.Canny(gray, 45, 130)
    edge_density = np.sum((edges > 0) & mask_bool) / (np.sum(mask_bool) + 1e-7)
    contour_count_ratio = min(len(valid_contours), 12) / 12.0

    moments = cv2.moments(largest)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)
    hu_moments = np.clip(hu_moments / 10.0, -1.0, 1.0)

    features = np.array([
        area_ratio,
        bbox_ratio,
        aspect_ratio,
        extent,
        perimeter_ratio,
        width_ratio,
        height_ratio,
        solidity,
        mean_gray,
        mean_saturation,
        mean_value,
        area_ratio * height_ratio,
        circularity,
        equivalent_diameter_ratio,
        contour_count_ratio,
        edge_density,
        center_x_ratio,
        center_y_ratio,
        *hu_moments
    ], dtype=np.float32)

    return features


def extract_features_from_image(image):
    cropped, mask = preprocess_image(image)

    hog_features = extract_hog_features(cropped)
    color_features = extract_color_histogram(cropped, mask)
    lbp_features = extract_lbp_features(cropped)
    shape_features = extract_shape_features(cropped, mask)

    combined = np.concatenate([
        hog_features,
        color_features,
        lbp_features,
        shape_features
    ])

    return combined.astype(np.float32)


def extract_features_from_path(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    return extract_features_from_image(image)


def extract_object_cues(image):
    cropped, mask = preprocess_image(image)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    mask_bool = mask > 0

    if np.sum(mask_bool) < 40:
        mask_bool = np.ones(mask.shape, dtype=bool)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    transparent_like = (
        (saturation < 65) &
        (value > 120) &
        mask_bool
    )

    blue_like = (
        (hue >= 85) &
        (hue <= 135) &
        (saturation > 45) &
        mask_bool
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    aspect_ratio = 1.0
    height_ratio = 0.0
    width_ratio = 0.0

    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 40
    ]

    if len(valid_contours) > 0:
        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        aspect_ratio = w / (h + 1e-7)
        height_ratio = h / IMAGE_SIZE[1]
        width_ratio = w / IMAGE_SIZE[0]

    cues = {
        "transparent_ratio": float(np.sum(transparent_like) / (np.sum(mask_bool) + 1e-7)),
        "blue_ratio": float(np.sum(blue_like) / (np.sum(mask_bool) + 1e-7)),
        "aspect_ratio": float(aspect_ratio),
        "height_ratio": float(height_ratio),
        "width_ratio": float(width_ratio)
    }

    return cues
