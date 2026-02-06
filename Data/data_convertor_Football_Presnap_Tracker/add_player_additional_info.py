import os, cv2, json
from tqdm import tqdm

merged_coco_path = "_annotations_roles.coco.json"
images_dir = "blured_images"
output_json = "_annotations_additional_info.coco.json"
source_file = "./Data/data_convertor_Football_Presnap_Tracker/Football Presnap Tracker.v1i.coco/merged_dataset"

additional_info_labels = ["position", "alignment"]
additional_info = {
    "position": {
        ord("1"): "RB",
        ord("2"): "WR",
        ord("3"): "TE",
    },
    "alignment": {
        ord("r"): "Right",
        ord("l"): "Left",
        ord("c"): "Center",
    },
}

QUIT_KEY = ord("q")
SKIP_KEY = ord("s")
CONFIRM_KEYS = [13, 10]

with open(os.path.join(source_file, merged_coco_path), "r") as f:
    coco = json.load(f)

annotations = coco["annotations"]
images = coco["images"]

anns_by_image = {}
for ann in annotations:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

print("Football Additional Info Labeling Tool\n")

def draw_player_info(img, ann, additional_info_labels):
    img_out = img.copy()
    x, y, w, h = map(int, ann["bbox"])
    cat = ann["category_id"]
    color = (0, 255, 0) if cat == 1 else (255, 0, 0) if cat == 3 else (0, 0, 255)

    cv2.rectangle(img_out, (x, y), (x + w, y + h), color, 2)

    info_text = " | ".join(f"{k}:{ann.get(k, '-')}" for k in additional_info_labels)
    cv2.putText(img_out, info_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(img_out, f"cat:{cat}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_out


# === Labeling Loop ===
for img in tqdm(images, desc="Labeling images"):
    img_path = os.path.join(source_file, images_dir, img["file_name"])
    if not os.path.exists(img_path):
        continue

    anns = anns_by_image.get(img["id"], [])
    original_image = cv2.imread(img_path)
    if original_image is None:
        continue

    for ann in anns:
        if ann["category_id"] in [0, 1, 4]:
            continue

        confirmed_player = False
        while not confirmed_player:
            disp = draw_player_info(original_image.copy(), ann, additional_info_labels)

            # Show instructions
            y0 = 25
            instructions = [
                "1–3: position | R/L/C: alignment",
                "S: skip player | Q: quit | Enter: confirm player",
            ]
            for line in instructions:
                cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y0 += 25

            cv2.imshow("Football Labeling Tool", disp)
            key = cv2.waitKey(0)

            if key == QUIT_KEY or key == 27:  # 'Q' or ESC
                print("\nExiting tool.")
                with open(os.path.join(source_file, output_json), "w") as f:
                    json.dump(coco, f, indent=2)
                cv2.destroyAllWindows()
                exit()

            elif key == SKIP_KEY:
                print("Skipped this player.")
                break

            elif key in CONFIRM_KEYS:
                confirmed_player = True
                print("Player confirmed.")

            else:
                # Update additional info
                for label in additional_info_labels:
                    if key in additional_info[label]:
                        ann[label] = additional_info[label][key]
                        print(f"{label.capitalize()} set: {additional_info[label][key]}")

    # === Auto-save progress after each image ===
    with open(os.path.join(source_file, output_json), "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Progress saved for image: {img['file_name']}")

cv2.destroyAllWindows()
print("\nAll images processed. Final annotations saved to:", output_json)
