import os, sys, json, cv2, torch, numpy as np, urllib.request, random
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_masks_auto.coco.json"
images_dir = os.path.join(source_dir, "images")
masks_dir = os.path.join(source_dir, "auto_masks")
os.makedirs(masks_dir, exist_ok=True)

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

with open(os.path.join(source_dir, coco_file), "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

img_to_anns = {}
for ann in anns:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)

def load_mask_png(mask_name, base_dir=masks_dir):
    path = os.path.join(base_dir, mask_name) if mask_name else None
    if path and os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (m > 127).astype(np.uint8)
    return None

def overlay_masks_on_image(image, ann_list, alpha=0.5):
    overlay = image.copy()
    for ann in ann_list:
        mask_name = ann.get("segmentation_mask")
        mask = load_mask_png(mask_name)
        if mask is None:
            continue
        color = tuple(int(c) for c in np.random.RandomState(ann["id"]).randint(0,255,3))
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask==1] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
    return overlay

def draw_bbox_and_id(img, ann, color=(0,255,0)):
    x, y, w, h = map(int, ann["bbox"])
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
    cv2.putText(img, f"id:{ann['id']}", (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_instructions(img, lines):
    h, w = img.shape[:2]
    box_h = 18 * len(lines) + 10
    cv2.rectangle(img, (6,6), (w-6, 6+box_h), (0,0,0), -1)
    alpha = 0.5
    sub = img.copy()
    cv2.addWeighted(sub, alpha, img, 1-alpha, 0, img)
    y = 6 + 18
    for ln in lines:
        cv2.putText(img, ln, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 18

image_ids = sorted(images.keys())
cur_idx = 0
edit_mode = False
cur_ann = None
cur_image = None
display_image = None
pos_points = []
neg_points = []
unsaved_mask = None
info_lines = []
painting = False
erase_mode = False
brush_radius = 10
paint_mask = None
WINDOW = "SAM QA - click player to edit (press q to quit)"

def find_ann_by_click(x, y, anns_for_image, max_dist=80):
    best = None
    bestd = max_dist
    for ann in anns_for_image:
        bx, by, bw, bh = ann["bbox"]
        cx, cy = bx + bw/2, by + bh/2
        d = np.hypot(cx - x, cy - y)
        if d < bestd:
            bestd = d
            best = ann
    return best

def mouse_cb(event, x, y, flags, param):
    global edit_mode, cur_ann, pos_points, neg_points, unsaved_mask
    global painting, erase_mode, paint_mask

    if edit_mode and painting:
        if event == cv2.EVENT_LBUTTONDOWN:
            erase_mode = False
            cv2.circle(paint_mask, (x, y), brush_radius, 1, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            erase_mode = True
            cv2.circle(paint_mask, (x, y), brush_radius, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON or flags & cv2.EVENT_FLAG_RBUTTON):
            color_val = 0 if erase_mode else 1
            cv2.circle(paint_mask, (x, y), brush_radius, color_val, -1)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if not edit_mode:
            anns_here = img_to_anns.get(cur_image_info["id"], [])
            ann = find_ann_by_click(x, y, anns_here)
            if ann:
                edit_mode = True
                cur_ann = ann
                pos_points = []
                neg_points = []
                unsaved_mask = None
                print(f"Entered edit mode for ann id {cur_ann['id']}")
        else:
            pos_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and edit_mode:
        neg_points.append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN and edit_mode:
        pos_points, neg_points, unsaved_mask = [], [], None

def resegment_current():
    global unsaved_mask, pos_points, neg_points, cur_image, cur_ann, predictor
    if cur_image is None or cur_ann is None:
        return None

    x, y, w, h = map(int, cur_ann["bbox"])
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(cur_image.shape[1], x2), min(cur_image.shape[0], y2)

    crop = cur_image[y:y2, x:x2]
    if crop.size == 0:
        print("Empty crop region.")
        return None

    predictor.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    pts, labels = [], []
    for p in pos_points:
        pts.append([p[0]-x, p[1]-y])
        labels.append(1)
    for p in neg_points:
        pts.append([p[0]-x, p[1]-y])
        labels.append(0)

    if len(pts) == 0:
        print("No points provided. Using box prompt inside bbox region.")
        masks, scores, _ = predictor.predict(box=np.array([0, 0, w, h]), multimask_output=True)
    else:
        pts_arr = np.array(pts)
        labels_arr = np.array(labels)
        masks, scores, _ = predictor.predict(point_coords=pts_arr, point_labels=labels_arr, multimask_output=True)

    if masks is None or len(masks)==0:
        print("SAM returned no masks.")
        return None

    idx = int(np.argmax(scores))
    mask_crop = masks[idx].astype(np.uint8)
    mask_full = np.zeros(cur_image.shape[:2], dtype=np.uint8)
    mask_full[y:y2, x:x2] = mask_crop
    unsaved_mask = mask_full
    print(f"Re-segmented ann {cur_ann['id']} (mask inside bbox {w}x{h})")
    return unsaved_mask

def save_current_mask():
    global unsaved_mask, cur_ann, paint_mask
    mask_to_save = unsaved_mask if unsaved_mask is not None else paint_mask
    if mask_to_save is None:
        print("No resegmented or manual mask to save.")
        return False
    imgfile = cur_image_info["file_name"]
    mask_name = f"{os.path.splitext(imgfile)[0]}_{cur_ann['id']}.png"
    out_path = os.path.join(masks_dir, mask_name)
    cv2.imwrite(out_path, (mask_to_save*255).astype(np.uint8))
    cur_ann["segmentation_mask"] = mask_name
    print(f"Saved mask to {out_path}")
    return True

def persist_coco():
    out_path = os.path.join(source_dir, coco_file)
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO JSON saved to {out_path}")

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, mouse_cb)

while True:
    cur_image_info = images[image_ids[cur_idx]]
    img_path = os.path.join(images_dir, cur_image_info["file_name"])
    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        cur_idx = (cur_idx + 1) % len(image_ids)
        continue

    cur_image = cv2.imread(img_path)
    anns_here = img_to_anns.get(cur_image_info["id"], [])
    display_image = overlay_masks_on_image(cur_image.copy(), anns_here, alpha=0.45)
    for ann in anns_here:
        draw_bbox_and_id(display_image, ann, color=(200,200,200))

    if not edit_mode:
        info_lines = [
            f"Image {cur_idx+1}/{len(image_ids)}: {cur_image_info['file_name']}",
            "Click near a player's bbox center to edit.",
            "Keys: n=next, p=prev, q=quit"
        ]
        draw_instructions(display_image, info_lines)
    else:
        display_image = overlay_masks_on_image(cur_image.copy(), anns_here, alpha=0.35)
        draw_bbox_and_id(display_image, cur_ann, color=(0,0,255))

        saved_mask = load_mask_png(cur_ann.get("segmentation_mask"))
        if saved_mask is not None:
            color_mask = np.zeros_like(display_image)
            color_mask[saved_mask==1] = (50,50,200)
            display_image = cv2.addWeighted(display_image, 1.0, color_mask, 0.25, 0)

        if unsaved_mask is not None:
            color_mask = np.zeros_like(display_image)
            color_mask[unsaved_mask==1] = (0,200,0)
            display_image = cv2.addWeighted(display_image, 1.0, color_mask, 0.5, 0)

        if paint_mask is not None:
            paint_overlay = np.zeros_like(display_image)
            paint_overlay[paint_mask==1] = (0,255,0)
            display_image = cv2.addWeighted(display_image, 1.0, paint_overlay, 0.4, 0)

        for p in pos_points:
            cv2.circle(display_image, (int(p[0]), int(p[1])), 6, (0,255,0), -1)
        for p in neg_points:
            cv2.circle(display_image, (int(p[0]), int(p[1])), 6, (0,0,255), -1)

        info_lines = [
            f"EDITING ann {cur_ann['id']}  ({cur_idx+1}/{len(image_ids)})",
            "Left: +point  Right: -point  Mid: clear  r=reseg  s=save  e=exit edit",
            "m=manual paint  n/p=next/prev  q=quit"
        ]
        draw_instructions(display_image, info_lines)

    cv2.imshow(WINDOW, display_image)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        print("Exiting.")
        break
    elif key == ord('n') and not edit_mode:
        cur_idx = (cur_idx + 1) % len(image_ids)
    elif key == ord('p') and not edit_mode:
        cur_idx = (cur_idx - 1) % len(image_ids)
    elif key == ord('e') and edit_mode:
        edit_mode = False
        cur_ann = None
        pos_points, neg_points, unsaved_mask, paint_mask = [], [], None, None
        print("Exited edit mode.")
    elif key == ord('r') and edit_mode:
        _ = resegment_current()
    elif key == ord('s') and edit_mode:
        if save_current_mask():
            persist_coco()
    elif key == ord('u') and edit_mode:
        if neg_points:
            neg_points.pop()
        elif pos_points:
            pos_points.pop()
        else:
            print("No points to undo.")
    elif key == ord('m') and edit_mode:
        if paint_mask is None:
            paint_mask = np.zeros(cur_image.shape[:2], np.uint8)
            saved_mask = load_mask_png(cur_ann.get("segmentation_mask"))
            if saved_mask is not None:
                paint_mask[:] = saved_mask
        painting = not painting
        print(f"Manual paint mode {'ON' if painting else 'OFF'}")

cv2.destroyAllWindows()
