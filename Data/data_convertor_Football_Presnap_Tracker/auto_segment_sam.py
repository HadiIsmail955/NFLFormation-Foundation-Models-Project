import os, sys, json, cv2, torch, numpy as np, urllib.request
from tqdm import tqdm
from collections import defaultdict
from segment_anything import SamPredictor, sam_model_registry
from Data.data_convertor_Football_Presnap_Tracker.utils.convert_mask_to_polygons import convert_mask_to_polygons

source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_all.coco.json"
images_dir = os.path.join(source_dir, "images")
masks_dir = os.path.join(source_dir, "auto_masks")
os.makedirs(masks_dir, exist_ok=True)
output_json = "_annotations_masks_auto.coco.json"

model_choice = input("Select SAM model (h=vit_h, l=vit_l, b=vit_b) [default h]: ").lower()
model_type = {"h": "vit_h", "l": "vit_l", "b": "vit_b"}.get(model_choice, "vit_h")
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_urls = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
sam_checkpoint = f"sam_{model_type}.pth"

if not os.path.exists(sam_checkpoint):
    url = sam_urls[model_type]
    print(f"Downloading SAM checkpoint ({model_type}) from {url} ...")
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = min(int(downloaded / total_size * 100), 100)
        sys.stdout.write(f"\rDownloading: {progress}%")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, sam_checkpoint, reporthook=show_progress)
    print("\nDownload complete!")

print(f"Using SAM {model_type} on {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

anns_by_image = defaultdict(list)
for ann in anns:
    anns_by_image[ann["image_id"]].append(ann)

print(f"Segmenting {len(anns)} annotations across {len(anns_by_image)} images...")

for image_id, ann_list in tqdm(anns_by_image.items()):
    img_info = images[image_id]
    img_path = os.path.join(images_dir, img_info["file_name"])

    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load: {img_path}")
        continue

    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for ann in ann_list:
        x, y, w, h = map(int, ann["bbox"])

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, image.shape[1] - 1)
        y2 = min(y + h, image.shape[0] - 1)

        masks, scores, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)].astype(np.uint8)

        mask_filename = f"{os.path.splitext(img_info['file_name'])[0]}_{ann['id']}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        cv2.imwrite(mask_path, best_mask * 255)

        polygons = convert_mask_to_polygons(best_mask)

        ann["segmentation"] = polygons
        ann["segmentation_mask"] = mask_filename
        ann["iscrowd"] = 0

print(f"All auto masks saved to {masks_dir}")

out_path = os.path.join(source_dir, output_json)
with open(out_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"Updated COCO JSON saved as {out_path}")
