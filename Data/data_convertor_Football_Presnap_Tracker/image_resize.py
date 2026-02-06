import os
import json
from tqdm import tqdm 
import numpy as np
from PIL import Image
from Data.data_convertor_Football_Presnap_Tracker.utils.resize_and_pad_image import resize_and_pad_image


source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_additional_info.coco.json"
image_dir = os.path.join(source_dir, "resize_images")
os.makedirs(image_dir, exist_ok=True)
resize_per_player_image_dir = os.path.join(source_dir, "resize_players_masks")
os.makedirs(resize_per_player_image_dir, exist_ok=True)
team_masks_dir = os.path.join(source_dir, "Team_masks")
os.makedirs(team_masks_dir, exist_ok=True)
off_team_masks_dir = os.path.join(team_masks_dir, "resize_off_masks")
os.makedirs(off_team_masks_dir, exist_ok=True)
def_team_masks_dir = os.path.join(team_masks_dir, "resize_def_masks")
os.makedirs(def_team_masks_dir, exist_ok=True)
all_team_masks_dir = os.path.join(team_masks_dir, "resize_all_masks")
os.makedirs(all_team_masks_dir, exist_ok=True)
off_team_multi_masks_dir = os.path.join(team_masks_dir, "resize_off_multi_masks")
os.makedirs(off_team_multi_masks_dir, exist_ok=True)
all_team_multi_masks_dir = os.path.join(team_masks_dir, "resize_all_multi_masks")
os.makedirs(all_team_multi_masks_dir, exist_ok=True)
output_json = "_annotations_final.coco.json"

playerMaskResize= False

with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)
if playerMaskResize:
    annotations=coco["annotations"]
    for annotation in tqdm(annotations):
        img_path = os.path.join(source_dir, "auto_masks", annotation["segmentation_mask"])
        image_np = np.array(Image.open(img_path).convert("RGB"))
        img_padded, meta = resize_and_pad_image(image_np, target_size=1024)

        save_name =  annotation["segmentation_mask"]
        save_path = os.path.join(resize_per_player_image_dir, save_name)
        img_padded.save(save_path)
else:
    images = coco["images"]
    files=[["team_mask",all_team_masks_dir,"Team_masks/all_masks"], ["offense_mask",off_team_masks_dir,"Team_masks/off_masks"],["team_multi_mask",all_team_multi_masks_dir,"Team_masks/all_multi_masks"], ["offense_multi_mask",off_team_multi_masks_dir,"Team_masks/off_multi_masks"], ["defense_mask",def_team_masks_dir,"Team_masks/def_masks"], ["file_name",image_dir,"images"]]


    for img_info  in tqdm(images):
        for file in files:
            img_path = os.path.join(source_dir, file[2],img_info[file[0]])
            image_np = np.array(Image.open(img_path).convert("RGB"))
            img_padded, meta = resize_and_pad_image(image_np, target_size=1024)

            save_name = img_info[file[0]]
            save_path = os.path.join(file[1], save_name)
            img_padded.save(save_path)

        img_info["resize_meta"] = meta

    with open(os.path.join(source_dir, output_json), 'w') as f:
        json.dump(coco, f)