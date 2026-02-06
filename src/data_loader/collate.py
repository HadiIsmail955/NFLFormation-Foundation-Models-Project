import torch

FIXED_KEYS = {
    "seg_image",
    "mask",
    "formation_label",
    "center_map",
    "position_masks", 
    "position_heatmaps",
}

def presnap_collate_fn(batch):
    collated = {}

    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if key in FIXED_KEYS:
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated
