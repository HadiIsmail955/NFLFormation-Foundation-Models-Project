import cv2

def convert_mask_to_polygons(best_mask):
    # Convert binary mask to COCO polygon format
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        c = c.flatten().tolist()
        if len(c) >= 6:  # valid polygon has at least 3 points
            polygons.append(c)
    return polygons