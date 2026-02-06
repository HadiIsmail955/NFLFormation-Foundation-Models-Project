import os
import math
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PresnapDataset(Dataset):
    RARE_FORMATIONS = ['pistol', 'victory', 'stacked', 'twins-left']

    ROLE_MAP = {
        "oline": 0,
        "qb": 1,
        "skill": 2
    }
    
    ALIGNMENT_MAP = {
         'Left':0, 
         'Pistol':1, 
         'Right':2, 
         'Shotgun':3, 
         'Under Center':4
    }

    # POSITION_MAP = {
    #         'FB':0,
    #         'QB':1,
    #         'RB':2, 
    #         'SB':3, 
    #         'TE':4, 
    #         'WB':5, 
    #         'WR':6,
    #     }

    POSITION_MERGE_MAP = {
        "FB": "RB",
        "RB": "RB",

        "SB": "WR",
        "WR": "WR",

        "WB": "TE",
        "TE": "TE",

        "QB": "QB",
    }

    POSITION_MAP = {
        "QB": 0,
        "RB": 1,
        "WR": 2,
        "TE": 3,
    }

    # FORMATION_MAP = {
    #     "shotgun": 0,
    #     "ace-left": 1,
    #     "ace-right": 2,
    #     "trips-left": 3,
    #     "trips-right": 4,
    #     "twins-right": 5,
    #     "bunch-left": 6,
    #     "bunch-right": 7,
    #     "i-formation": 8,
    #     "trey-left": 9,
    #     "trey-right": 10,
    #     "empty": 11,
    #     "double-tight": 12,
    #     "heavy": 13,
    # }

    FLIP_MAP = {
        "trips-left": "trips-right",
        "trips-right": "trips-left",
        "bunch-left": "bunch-right",
        "bunch-right": "bunch-left",
        "trey-left": "trey-right",
        "trey-right": "trey-left",
        "ace-left": "ace-right",
        "ace-right": "ace-left",
    }

    FORMATION_MERGE_MAP = {
        "trips-left": "trips",
        "trips-right": "trips",
        "bunch-left": "bunch",
        "bunch-right": "bunch",
        "trey-left": "trey",
        "trey-right": "trey",
        "ace-left": "ace",
        "ace-right": "ace",
        "twins-right": "twins",
    }
    FORMATION_MAP = {
        "shotgun": 0,
        "empty": 1,
        "i-formation": 2,
        "double-tight": 3,
        "heavy": 4,
        "twins": 5,
        "trips": 6,
        "bunch": 7,
        "trey": 8,
        "ace": 9,
    }

    def __init__(self, data_source, coco_file, seg_transform=None, classifier_transform=None, enable_flip=False, flip_prob=0.5):
        super().__init__()
        self.data_source=data_source
        self.seg_transform=seg_transform
        self.classifier_transform=classifier_transform
        self.enable_flip=enable_flip
        self.flip_prob=flip_prob
        with open(coco_file, 'r') as f:
            self.coco = json.load(f)
        from collections import Counter

        labels = [img["attributes"]["formation"] for img in self.coco["images"]
                if img.get("attributes", {}).get("formation") in self.FORMATION_MAP]
        
        print(Counter(labels))

        labels = []
        for img in self.coco["images"]:
            raw = img.get("attributes", {}).get("formation")
            if raw is None:
                continue

            merged = self.FORMATION_MERGE_MAP.get(raw, raw)

            if merged in self.FORMATION_MAP:
                labels.append(merged)

        print(Counter(labels))

        self.img_folder_path = os.path.join(data_source,"images")
        self.seg_img_folder_path = os.path.join(data_source,"resize_images")
        self.mask_img_folder_path = os.path.join(data_source,"Team_masks","resize_off_masks")
        self.mask_per_player_img_folder_path = os.path.join(data_source,"resize_players_masks")
        self.images = [
            img for img in self.coco['images']
            if img.get('attributes', {}).get('formation') in self.FORMATION_MAP
        ]
        
        self.images = []
        for img in self.coco["images"]:
            raw = img.get("attributes", {}).get("formation")
            if raw is None:
                continue

            merged = self.FORMATION_MERGE_MAP.get(raw, raw)

            if merged in self.FORMATION_MAP:
                self.images.append(img)

        filtered_images = []

        for img in self.images:
            anns = [
                ann for ann in self.coco.get("annotations", [])
                if ann["image_id"] == img["id"]
            ]

            num_oline = 0
            num_qb = 0
            num_skill_valid = 0
            num_skill_invalid = 0

            for ann in anns:
                cat_id = ann["category_id"]
                cat_name = next(
                    (c["name"] for c in self.coco["categories"] if c["id"] == cat_id),
                    None,
                )

                if cat_name == "oline":
                    num_oline += 1

                elif cat_name == "qb":
                    num_qb += 1

                # elif cat_name == "skill":
                #     pos = ann.get("position", None)
                #     if pos is not None and pos.strip().upper() in self.POSITION_MAP:
                #         num_skill_valid += 1
                #     else:
                #         num_skill_invalid += 1
                elif cat_name == "skill":
                    raw_pos = ann.get("position", None)
                    if raw_pos is not None:
                        raw_pos = raw_pos.strip().upper()
                        merged_pos = self.POSITION_MERGE_MAP.get(raw_pos, None)
                        if merged_pos in self.POSITION_MAP:
                            num_skill_valid += 1
                        else:
                            num_skill_invalid += 1
                    else:
                        num_skill_invalid += 1

            num_players_valid = num_oline + num_qb + num_skill_valid

            if num_players_valid >= 11:
                filtered_images.append(img)

        self.images = filtered_images
        print(f"Total images after filtering: {len(self.images)}")



    @staticmethod
    def generate_center_heatmap(centers, H, W, sigma=3):
        H = int(H)
        W = int(W)
        heatmap = torch.zeros((H, W), dtype=torch.float32)

        if len(centers) == 0:
            return heatmap

        radius = int(3 * sigma)

        for (cx, cy) in centers:
            cx = int(round(float(cx)))
            cy = int(round(float(cy)))

            x0 = max(0, cx - radius)
            x1 = min(W, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(H, cy + radius + 1)

            for y in range(y0, y1):
                for x in range(x0, x1):
                    heatmap[y, x] += math.exp(
                        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)
                    )

        heatmap.clamp_(0, 1)
        return heatmap

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        do_flip = self.enable_flip and random.random() < self.flip_prob

        img_entry = self.images[idx]

        # img_path = os.path.join(self.img_folder_path, img_entry['file_name'])
        img_path = os.path.join(self.seg_img_folder_path, img_entry['file_name'])
        image = Image.open(img_path).convert("RGB")

        seg_img_path= os.path.join(self.seg_img_folder_path, img_entry['file_name'])
        seg_image = Image.open(seg_img_path).convert("RGB")

        mask_path = os.path.join(self.mask_img_folder_path, img_entry['offense_mask'])
        mask=  Image.open(mask_path).convert("L")



        if self.seg_transform:
            seg_image, mask = self.seg_transform.apply(
                image=seg_image,
                mask=mask,
                do_flip=do_flip,
            )
        if self.classifier_transform:
            image= self.classifier_transform(image)
        else:
            image = transforms.ToTensor()(image)

        meta=img_entry["resize_meta"]
        
        # formation_str = img_entry.get('attributes', {}).get('formation', 'unknown')
        # formation_name = formation_str if formation_str in self.FORMATION_MAP else 'unknown'
        # formation_label = self.FORMATION_MAP.get(formation_str, -1)
        # formation_label = torch.tensor(formation_label, dtype=torch.long)

        formation_raw = img_entry.get('attributes', {}).get('formation', 'unknown')
        formation_merged = self.FORMATION_MERGE_MAP.get(
            formation_raw,
            formation_raw
        )
        formation_name = formation_merged if formation_merged in self.FORMATION_MAP else 'unknown'
        formation_label = self.FORMATION_MAP.get(formation_merged, -1)
        formation_label = torch.tensor(formation_label, dtype=torch.long)

        bboxes = []
        roles = []
        positions = []
        alignments = []
        playerMasks = []
        centers = []
        # heatmap_points = {
        #     "FB": [],
        #     "QB": [],
        #     "RB": [],
        #     "SB": [],
        #     "TE": [],
        #     "WB": [],
        #     "WR": [],
        #     "OLINE": [],
        # }

        heatmap_points = {
            "QB": [],
            "RB": [],
            "WR": [],
            "TE": [],
            "OLINE": [],
        }

        count_masks = 0
        
        position_masks = {
            pos: torch.zeros_like(mask)
            for pos in self.POSITION_MAP.values()
        }
        position_points = {
            pos: [] for pos in self.POSITION_MAP.values()
        }
        oline_mask = torch.zeros_like(mask)

        anns = [ann for ann in self.coco.get('annotations', []) if ann['image_id'] == img_entry['id']]
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = next(
                (c["name"] for c in self.coco["categories"] if c["id"] == cat_id),
                None,
            )

            if cat_name not in self.ROLE_MAP:
                continue
            
            role_id = self.ROLE_MAP.get(cat_name, -1)
            roles.append(role_id)

            scale = meta['scale']
            pad_left = meta['pad_left']
            pad_top = meta['pad_top']
            new_w = meta['new_w']
            new_h = meta['new_h']
            
            x, y, w, h = ann['bbox']
            
            x_new = x * scale + pad_left
            y_new = y * scale + pad_top
            w_new = w * scale
            h_new = h * scale
            
            x_norm = x_new / new_w
            y_norm = y_new / new_h
            w_norm = w_new / new_w
            h_norm = h_new / new_h

            if do_flip:
                x_norm = 1.0 - (x_norm + w_norm)
            
            bboxes.append([x_norm, y_norm, w_norm, h_norm])
            
            # position_str = ann.get("position", None)
            # position_label = self.POSITION_MAP.get(position_str, -1)
            # positions.append(position_label)

            raw_pos = ann.get("position", None)
            if raw_pos is not None:
                raw_pos = raw_pos.strip().upper()
                merged_pos = self.POSITION_MERGE_MAP.get(raw_pos, None)
            else:
                merged_pos = None
            position_label = self.POSITION_MAP.get(merged_pos, -1)
            positions.append(position_label)

            alignment_str = ann.get("alignment", None)
            alignment_label = self.ALIGNMENT_MAP.get(alignment_str, -1)
            alignments.append(alignment_label)

            mask_per_player_path = os.path.join(self.mask_per_player_img_folder_path, ann["segmentation_mask"])
            mask_per_player = Image.open(mask_per_player_path).convert("L")
            _, pm = self.seg_transform.apply(
                image=None,
                mask=mask_per_player,
                do_flip=do_flip,
            )
            playerMasks.append(pm)

            # if position_str is not None:
            #     pos_key = position_str.strip().upper()
            #     position_label = self.POSITION_MAP.get(pos_key, -1)
            if merged_pos is not None:
                pos_key = merged_pos.strip().upper()
                position_label = self.POSITION_MAP.get(pos_key, -1)
            else:
                position_label = -1

            if position_label >= 0:
                position_masks[position_label] = torch.maximum(
                    position_masks[position_label], pm
                )
                count_masks += 1

            if cat_name is not None and cat_name.lower() == "oline":
                oline_mask = torch.maximum(oline_mask, pm)
                count_masks += 1

            # pm2 = pm.squeeze(0)
            # ys, xs = torch.where(pm2 > 0)
            # if len(xs) == 0:
            #     continue
            # y_min = ys.min()
            # band = (ys <= y_min + 3)  
            # cx = xs[band].float().mean()
            # cy = ys[band].float().mean()
            # centers.append((cx, cy))

            pm2 = pm.squeeze(0)
            ys, xs = torch.where(pm2 > 0)
            if len(xs) == 0:
                continue
            head_percentile = 0.18  
            k = max(1, int(len(ys) * head_percentile))
            sorted_ys, idx = torch.sort(ys)
            cy = sorted_ys[:k].float().mean()
            cx = xs[idx[:k]].float().mean()
            centers.append((cx, cy))

            # pm2 = pm.squeeze(0)
            # ys, xs = torch.where(pm2 > 0)
            # if len(xs) == 0:
            #     continue
            # y_min = ys.min().float()
            # y_max = ys.max().float()
            # h = y_max - y_min + 1
            # head_to_jersey_start = y_min + 0.10 * h   
            # head_to_jersey_end   = y_min + 0.50 * h  
            # band = (ys >= head_to_jersey_start) & (ys <= head_to_jersey_end)
            # if band.sum() == 0:
            #     band = ys <= (y_min + 0.25 * h)
            # cx = xs[band].float().mean()
            # cy = ys[band].float().mean()
            # centers.append((cx, cy))

            pm2 = pm.squeeze(0)
            ys, xs = torch.where(pm2 > 0)

            k = max(1, int(len(ys) * 0.18))
            sorted_ys, idx = torch.sort(ys)
            cy = sorted_ys[:k].float().mean()
            cx = xs[idx[:k]].float().mean()

            if cat_name == "oline":
                heatmap_points["OLINE"].append((cx.item(), cy.item()))

            elif cat_name == "qb":
                heatmap_points["QB"].append((cx.item(), cy.item()))

            # elif cat_name == "skill":
            #     pos_key = ann.get("position", "").strip().upper()
            #     if pos_key in heatmap_points:
            #         heatmap_points[pos_key].append((cx.item(), cy.item()))
            elif cat_name == "skill":
                raw_pos = ann.get("position", "").strip().upper()
                merged_pos = self.POSITION_MERGE_MAP.get(raw_pos, None)
                if merged_pos in heatmap_points:
                    heatmap_points[merged_pos].append((cx.item(), cy.item()))

        
        points_label = torch.ones(len(centers), dtype=torch.int64)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        roles = torch.tensor(roles, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
        alignments = torch.tensor(alignments, dtype=torch.long)
        
        H, W = mask.shape[1:]
        center_map = self.generate_center_heatmap(centers, H, W).unsqueeze(0)

        # HEATMAP_ORDER = ["FB","QB","RB","SB","TE","WB","WR","OLINE"]
        HEATMAP_ORDER = ["QB", "RB", "WR", "TE", "OLINE"]

        position_heatmaps = torch.stack(
            [
                self.generate_center_heatmap(
                    heatmap_points[name], H, W, sigma=4
                )
                for name in HEATMAP_ORDER
            ],
            dim=0
        )


        position_masks_tensor = torch.stack(
            [position_masks[pos_id] for pos_id in sorted(self.POSITION_MAP.values())],
            dim=0
        ).squeeze(1)

        bg = (position_masks_tensor.sum(dim=0) == 0).float()

        position_masks_tensor = torch.cat(
            [
                bg.unsqueeze(0),        
                position_masks_tensor,  
                oline_mask
            ],
            dim=0
        )

        position_heatmaps = torch.cat(
            [
                bg.unsqueeze(0),
                position_heatmaps,
            ],
            dim=0
        )

        return {
                "image": image,
                "seg_image": seg_image,
                "mask": mask,
                "meta": meta,
                "formation_label": formation_label,
                "formation_name": formation_name,
                "bboxes": bboxes,
                "roles": roles,
                "positions": positions,
                "alignments": alignments,
                "playerMasks": playerMasks,
                "centers": centers,
                "points_label": points_label,
                "center_map": center_map,
                "position_masks": position_masks_tensor, 
                "position_heatmaps": position_heatmaps,
                "num_masks": count_masks,
                "file_name": img_entry['file_name'],
            }
        