import cv2
import numpy as np

def merge_team_masks(mask_files, output_path=None):
    merged = None

    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = (mask > 128).astype(np.uint8)

        if merged is None:
            merged = mask
        else:
            merged = np.maximum(merged, mask) 

    if merged is not None and output_path is not None:
        cv2.imwrite(output_path, merged * 255)
        # print("Saved merged mask:", output_path)


def merge_team_masks_color(offense_mask_path, defense_mask_path, output_path=None,
                           offense_color=(255, 0, 0),
                           defense_color=(0, 0, 255),
                           overlap_color=None):

    offense = cv2.imread(offense_mask_path, cv2.IMREAD_GRAYSCALE)
    defense = cv2.imread(defense_mask_path, cv2.IMREAD_GRAYSCALE)

    if offense is None or defense is None:
        raise ValueError("One of the team mask files could not be read.")

    offense = (offense > 128).astype(np.uint8)
    defense = (defense > 128).astype(np.uint8)
    h, w = offense.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    overlap = (offense & defense).astype(bool)
    color_mask[defense == 1] = defense_color
    color_mask[offense == 1] = offense_color

    if overlap_color is not None:
        color_mask[overlap] = overlap_color

    if output_path is not None:
        cv2.imwrite(output_path, color_mask)
        # print(f"Saved merged color team mask: {output_path}")

def merge_team_masks_color_extended(mask_paths_dict, output_path=None,
                                    subtype_colors=None,
                                    defense_color=(0, 0, 255),
                                    overlap_color=None):
    if subtype_colors is None:
        subtype_colors = {"oline": (255,0,0), "qb": (0,255,0), "skill": (0,255,0)}
    
    # Determine mask size from first available mask
    all_masks = [p for paths in mask_paths_dict.values() for p in paths]
    first_mask = cv2.imread(all_masks[0], cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        raise ValueError("No valid mask found to determine size.")
    h, w = first_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Merge defense first
    for def_mask_path in mask_paths_dict.get("defense", []):
        mask = cv2.imread(def_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 128).astype(bool)
        color_mask[mask] = defense_color

    # Merge offense subtypes
    for subtype, paths in mask_paths_dict.items():
        if subtype == "defense":
            continue
        color = subtype_colors.get(subtype, (255, 255, 255))
        for path in paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 128).astype(bool)
            color_mask[mask] = color

    # Handle overlap if specified
    if overlap_color is not None:
        overlap = np.zeros((h, w), dtype=bool)
        for paths in mask_paths_dict.values():
            for path in paths:
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                overlap |= (mask > 128)
        color_mask[overlap] = overlap_color

    if output_path is not None:
        cv2.imwrite(output_path, color_mask)
