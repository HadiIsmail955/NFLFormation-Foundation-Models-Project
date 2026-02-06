import os, sys, json, torch, numpy as np, urllib.request
from tqdm import tqdm
from collections import defaultdict
from Data.data_convertor_Football_Presnap_Tracker.utils.merge_masks import merge_team_masks, merge_team_masks_color, merge_team_masks_color_extended

source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_masks_auto.coco.json"
masks_dir = os.path.join(source_dir, "auto_masks")
team_masks_dir = os.path.join(source_dir, "Team_masks")
os.makedirs(team_masks_dir, exist_ok=True)
off_team_masks_dir = os.path.join(team_masks_dir, "off_masks")
os.makedirs(off_team_masks_dir, exist_ok=True)
off_team_multi_masks_dir = os.path.join(team_masks_dir, "off_multi_masks")
os.makedirs(off_team_multi_masks_dir, exist_ok=True)
def_team_masks_dir = os.path.join(team_masks_dir, "def_masks")
os.makedirs(def_team_masks_dir, exist_ok=True)
all_team_masks_dir = os.path.join(team_masks_dir, "all_masks")
os.makedirs(all_team_masks_dir, exist_ok=True)
all_team_multi_masks_dir = os.path.join(team_masks_dir, "all_multi_masks")
os.makedirs(all_team_multi_masks_dir, exist_ok=True)
output_json = "_annotations_mergered_masks_auto.coco.json"

offense_types = {
    2: "oline",
    3: "qb",
    5: "skill"
}
subtype_colors = {
    "oline": (255, 0, 0),   
    "qb": (0, 255, 0),      
    "skill": (0, 0, 255)    
}
defense_color = (0, 255, 255) 

with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

anns_by_image = defaultdict(list)
for ann in anns:
    anns_by_image[ann["image_id"]].append(ann)
    
for image_id, ann_list in tqdm(anns_by_image.items()):
    team_masks = {"offense": [], "defense": []}
    mask_paths = {
        "oline": [],
        "qb": [],
        "skill": [],
        "defense": []
    }
    img_info = images[image_id]
    for ann in ann_list:
        cat = ann["category_id"]
        if cat in [2,3,5]:  # Offense
            team_masks["offense"].append(os.path.join(masks_dir, ann["segmentation_mask"]))
            subtype = offense_types[cat]
            mask_paths[subtype].append(os.path.join(masks_dir, ann["segmentation_mask"]))
        elif cat in [1]:  # Defense
            team_masks["defense"].append(os.path.join(masks_dir, ann["segmentation_mask"]))
            mask_paths["defense"].append(os.path.join(masks_dir, ann["segmentation_mask"]))

    image_name=img_info["file_name"]
    team_mask_paths = os.path.join(all_team_masks_dir, f"{image_name}_team_mask.png")
    off_team_masks_paths = os.path.join(off_team_masks_dir, f"{image_name}_off_mask.png")
    def_team_masks_paths = os.path.join(def_team_masks_dir, f"{image_name}_def_mask.png")
    off_team_multi_masks_paths = os.path.join(off_team_multi_masks_dir, f"{image_name}_off_multi_mask.png")
    team_multi_mask_paths = os.path.join(all_team_multi_masks_dir, f"{image_name}_team_multi_mask.png")

    merge_team_masks(team_masks["offense"], output_path=off_team_masks_paths)
    merge_team_masks(team_masks["defense"], output_path=def_team_masks_paths)

    merge_team_masks_color(
        offense_mask_path=off_team_masks_paths,
        defense_mask_path=def_team_masks_paths,
        output_path=team_mask_paths
    )

    merge_team_masks_color_extended(mask_paths,off_team_multi_masks_paths,subtype_colors,(0,0,0))
    merge_team_masks_color_extended(mask_paths,team_multi_mask_paths,subtype_colors,defense_color)

    images[image_id]["team_mask"] = f"{image_name}_team_mask.png"
    images[image_id]["offense_mask"] = f"{image_name}_off_mask.png"
    images[image_id]["defense_mask"] = f"{image_name}_def_mask.png"
    images[image_id]["team_multi_mask"] = f"{image_name}_team_multi_mask.png"
    images[image_id]["offense_multi_mask"] = f"{image_name}_off_multi_mask.png"
    
with open(os.path.join(source_dir, output_json), 'w') as f:
    json.dump(coco, f)
print("Merged team masks saved and COCO JSON updated.")
    

        