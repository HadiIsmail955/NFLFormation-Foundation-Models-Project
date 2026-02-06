
import torch
import torch.nn as nn
import copy

def convert_sam_to_multi_class_decoder(sam, num_classes):
    md = sam.mask_decoder
    d = md.transformer_dim

    md.num_multimask_outputs = num_classes
    md.num_mask_tokens = num_classes + 1 

    md.mask_tokens = nn.Embedding(md.num_mask_tokens, d)

    md.output_hypernetworks_mlps = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d // 8),
        )
        for _ in range(md.num_mask_tokens)
    ])

    md.iou_prediction_head = nn.Sequential(
        nn.Linear(d, d),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d, md.num_mask_tokens),
    )

    return sam
