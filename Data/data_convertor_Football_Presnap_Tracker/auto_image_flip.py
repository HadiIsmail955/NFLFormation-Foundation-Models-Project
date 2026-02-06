import json
import os
import cv2
import shutil
from tqdm import tqdm
from Data.data_convertor_Football_Presnap_Tracker.utils.compare_position import compare_position
from Data.data_convertor_Football_Presnap_Tracker.utils.flip_image import flip_image_horizontally

source_dir = "./Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco/merged_dataset"
input_json_name = "_annotations_all.coco.json"
output_json_name = "_annotations_all.coco.json"

categories = None

coco_path = os.path.join(source_dir, input_json_name)

if not os.path.exists(coco_path):
    print(f"Missing: {coco_path}, skipping this split.")
    exit(1)

with open(coco_path, "r") as f:
    coco = json.load(f)

if categories is None:
    categories = coco["categories"]

images = coco["images"]
annotations = coco.get("annotations", [])

for img in tqdm(images, desc=f"Reviewing images"):
    img_path = os.path.join(source_dir,"images", img["file_name"])
    if not os.path.exists(img_path):
        print(f"Missing file: {img_path}")
        continue
    allPlayers=[ann for ann in annotations if ann["image_id"] == img["id"]]
    players = [ann for ann in allPlayers if ann["category_id"] in [1,3]]
    if len(players) == 0:
        print(f"No players found in image: {img_path}, skipping flip.")
        continue
    
    players_qb=[player for player in players if player["category_id"]==3]
    players_defense=[player for player in players if player["category_id"]==1]

    flip_count=0
    total_comparisons=0
    for player in players_qb:
        total_comparisons+=len(players_defense)
        for player_d in players_defense:
            if compare_position(player,player_d)==1:
                flip_count+=1
    
    if total_comparisons==0:
        print(f"No valid player comparisons in image: {img_path}, skipping flip.")
        continue

    flip_ratio=flip_count/total_comparisons
    if flip_ratio>0.6:
        print(f"Flipping image: {img_path} (Flip Ratio: {flip_ratio:.2f})")
        flipped_anns = flip_image_horizontally(image_path=img_path, anns=allPlayers)
        for flipped_ann in flipped_anns:
            for i, ann in enumerate(annotations):
                if ann["id"] == flipped_ann["id"]:
                    annotations[i] = flipped_ann
        
output_json_path = os.path.join(source_dir, output_json_name)
coco["annotations"] = annotations
with open(output_json_path, "w") as f:
    json.dump(coco, f, indent=4)

print(f"Updated annotations saved to: {output_json_path}")



