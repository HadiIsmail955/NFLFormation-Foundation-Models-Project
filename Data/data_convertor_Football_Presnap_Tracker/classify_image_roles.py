import os, cv2, json
from tqdm import tqdm

merged_coco_path = "_annotations_mergered_masks_auto_resized.coco.json"
images_dir = "images"
output_json = "_annotations_roles.coco.json"
source_file = "./Data/data_convertor_Football_Presnap_Tracker/Football Presnap Tracker.v1i.coco/merged_dataset"
labeled_dir = os.path.join(source_file, "blured_images")
os.makedirs(labeled_dir, exist_ok=True)

blur_unknown = True
UNBLUR_KEY = ord("u")   
FLIP_KEY = ord("f")     

SKIP_KEY = ord("s")
QUIT_KEY = ord("q")
CONFIRM_KEYS = [13, 32]  

formations = {
    ord("1"): "shotgun",
    ord("2"): "i-formation",
    ord("3"): "singleback",
    ord("4"): "trips-right",
    ord("5"): "trips-left",
    ord("6"): "empty",
    ord("7"): "pistol"
}

with open(os.path.join(source_file, merged_coco_path), "r") as f:
    coco = json.load(f)

annotations = coco["annotations"]
images = coco["images"]

anns_by_image = {}
for ann in annotations:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

print("Football Labeling Tool — with MIRROR FLIP (horizontal) support")

def draw_boxes(base_img, anns):
    img_out = base_img.copy()
    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        cat = ann["category_id"]
        color = (0,255,0) if cat==1 else (255,0,0) if cat==3 else (0,0,255) if cat==5 else (255,0,255) if cat==2 else (128,128,128) 
        cv2.rectangle(img_out, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_out, f"cat:{cat}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_out

def flip_bboxes_horizontally(anns, img_width):
    new_anns = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        new_x = img_width - x - w
        ann["bbox"] = [int(new_x), int(y), int(w), int(h)]
        new_anns.append(ann)
    return new_anns

updated_images = []

for img in tqdm(images):
    img_path = os.path.join(source_file, images_dir, img["file_name"])
    if not os.path.exists(img_path):
        continue

    anns = anns_by_image.get(img["id"], [])
    original_image = cv2.imread(img_path)
    if original_image is None:
        continue

    H, W = original_image.shape[:2]
    display_image = original_image.copy()
    save_image = original_image.copy()

    if blur_unknown:
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            if ann["category_id"] in [0,4]:
                roi = save_image[y:y+h, x:x+w]
                save_image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (23,23), 30)

    display_image = draw_boxes(display_image, anns)

    flipped = False
    formation = None
    confirmed = False
    unblur = False

    while True:
        disp = display_image.copy()
        y0 = 25
        instructions = [
            "1–7: formation | F: flip horizontally | U: unblur",
            "S: skip | Q: quit | Enter: confirm"
        ]
        for line in instructions:
            cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y0 += 30

        cv2.putText(disp, f"Flipped: {'Yes' if flipped else 'No'}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        status = f"Formation: {formation or '-'}"
        cv2.putText(disp, status, (10, y0+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Football Labeling Tool", disp)
        key = cv2.waitKey(0)

        if key == QUIT_KEY:
            print("Exiting tool.")
            cv2.destroyAllWindows()
            exit()
        elif key == SKIP_KEY:
            print("Skipped image.")
            break
        elif key in formations:
            formation = formations[key]
            print(f"Formation: {formation}")
        elif key == UNBLUR_KEY:
            unblur = True
            print("Unblur activated.")
        elif key == FLIP_KEY:
            print("Flipping image horizontally...")
            display_image = cv2.flip(display_image, 1)
            save_image = cv2.flip(save_image, 1)
            original_image = cv2.flip(original_image, 1)
            anns = flip_bboxes_horizontally(anns, W)
            display_image = draw_boxes(display_image, anns)
            flipped = not flipped
        elif key in CONFIRM_KEYS:
            if formation:
                confirmed = True
                break
            else:
                print("Please select formation first.")

    if confirmed:
        img["attributes"] = {
            "formation": formation,
            "flipped": flipped
        }
        updated_images.append(img)

        final_img = original_image.copy() if unblur else save_image
        save_path = os.path.join(labeled_dir, img["file_name"])
        cv2.imwrite(save_path, final_img)

cv2.destroyAllWindows()

# Save final JSON
coco["images"] = updated_images
with open(os.path.join(source_file, output_json), "w") as f:
    json.dump(coco, f, indent=2)

print(f"\nSaved dataset with mirror-flip bounding boxes → {output_json}")
print(f"Labeled images saved to: {labeled_dir}")
