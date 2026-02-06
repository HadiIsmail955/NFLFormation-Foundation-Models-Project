import cv2
def flip_bboxes_horizontally(anns, img_width):
    new_anns = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        new_x = img_width - x - w
        ann["bbox"] = [int(new_x), int(y), int(w), int(h)]
        new_anns.append(ann)
    return new_anns

def flip_image_horizontally(image_path, anns, output_path=None,overwrite=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    flipped_image = cv2.flip(image, 1)
    H, W = image.shape[:2]
    flipped_anns = flip_bboxes_horizontally(anns, W)
    if overwrite:
        cv2.imwrite(image_path, flipped_image)
    if output_path is not None:
        cv2.imwrite(output_path, flipped_image)
    return flipped_anns