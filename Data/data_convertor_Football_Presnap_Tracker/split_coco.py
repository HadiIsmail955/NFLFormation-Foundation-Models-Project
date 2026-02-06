import json
import os
import random
from copy import deepcopy
from collections import defaultdict

source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_additional_info.coco.json"
output_dir = "splits"
train_ratio = 0.9
val_ratio = 0.0
test_ratio = 0.1
random_seed = 42

output_dir=os.path.join(source_dir,output_dir)
os.makedirs(output_dir, exist_ok=True)

random.seed(random_seed)

coco_path=os.path.join(source_dir,coco_file)
with open(coco_path, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco.get("annotations", [])

formation_to_images = defaultdict(list)

for img in images:
    formation = img.get("attributes", None).get("formation", None)
    if formation is None:
        formation = "unknown"  
    formation_to_images[formation].append(img)


train_images, val_images, test_images = [], [], []

for formation, imgs in formation_to_images.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val 

    train_images.extend(imgs[:n_train])
    val_images.extend(imgs[n_train:n_train + n_val])
    test_images.extend(imgs[n_train + n_val:])

def filter_annotations(images_list, annotations):
    image_ids = set(img["id"] for img in images_list)
    return [ann for ann in annotations if ann["image_id"] in image_ids]

train_annotations = filter_annotations(train_images, annotations)
val_annotations = filter_annotations(val_images, annotations)
test_annotations = filter_annotations(test_images, annotations)

def build_coco_dict(images_list, annotations_list):
    new_coco = deepcopy(coco)
    new_coco["images"] = images_list
    new_coco["annotations"] = annotations_list
    return new_coco

train_coco = build_coco_dict(train_images, train_annotations)
val_coco = build_coco_dict(val_images, val_annotations)
test_coco = build_coco_dict(test_images, test_annotations)

with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_coco, f, indent=4)

with open(os.path.join(output_dir, "val.json"), "w") as f:
    json.dump(val_coco, f, indent=4)

with open(os.path.join(output_dir, "test.json"), "w") as f:
    json.dump(test_coco, f, indent=4)

print("Split completed!")
print(f"Train: {len(train_images)} images")
print(f"Validation: {len(val_images)} images")
print(f"Test: {len(test_images)} images\n")

for formation, imgs in formation_to_images.items():
    n_train = len([img for img in train_images if img.get("attributes", {}).get("formation", "unknown") == formation])
    n_val = len([img for img in val_images if img.get("attributes", {}).get("formation", "unknown") == formation])
    n_test = len([img for img in test_images if img.get("attributes", {}).get("formation", "unknown") == formation])
    
    print(f"{formation} - Train: {n_train}, Val: {n_val}, Test: {n_test}")