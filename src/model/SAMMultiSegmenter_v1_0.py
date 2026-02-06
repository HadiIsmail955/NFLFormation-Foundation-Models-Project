import os
import sys
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .backbone.sam_backbone import SAMBackbone
from .prompt.sam_prompt_encoder import SAMPromptEncoder
from .decoder.sam_mask_decoder import SAMMaskDecoder
from .decoder.convert_sam_to_multi_class_decoder import convert_sam_to_multi_class_decoder
from .head.classification_head import FormationHead

class SAMSegmenter(nn.Module):
    def __init__(self, sam_type="vit_h", ckpt_dir="./model/models", unfreeze_last_blocks=0,num_classes=7, num_formations=14):
        super().__init__()

        sam_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        
        sam_checkpoint = f"sam_{sam_type}.pth"
        
        ckpt_root = ckpt_dir
        os.makedirs(ckpt_root, exist_ok=True)

        ckpt_path = os.path.join(ckpt_root, sam_checkpoint)

        if not os.path.exists(ckpt_path):
            url = sam_urls[sam_type]
            print(f"Downloading SAM checkpoint ({sam_type}) from {url} ...")

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = min(int(downloaded / total_size * 100), 100)
                sys.stdout.write(f"\rDownloading: {progress}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, ckpt_path, reporthook=show_progress)
            print("\nDownload complete!")

        sam = sam_model_registry[sam_type](checkpoint=ckpt_path)

        sam = convert_sam_to_multi_class_decoder(sam, num_classes)

        self.backbone = SAMBackbone(
            sam=sam,
            unfreeze_last_blocks=unfreeze_last_blocks
        )

        self.prompt = SAMPromptEncoder(sam)
        self.decoder = SAMMaskDecoder(sam)
        
        EMB_DIM = 256 
        self.formation_head = FormationHead(
            EMB_DIM + num_classes,
            num_formations
        )

    def forward(self, x, predict_formation=False,multimask_output=False):
        img_emb = self.backbone(x)                    # [B,256,h,w]
        sparse_emb, dense_emb = self.prompt()

        masks = self.decoder(
            image_embeddings=img_emb,
            sparse_emb=sparse_emb,
            dense_emb=dense_emb,
            out_size=x.shape[-2:],
            multimask_output=multimask_output
        )                                             # [B,P,H,W]

        if not predict_formation:
            return masks

        masks_ds = F.interpolate(
                masks,                       
                size=img_emb.shape[-2:],     
                mode="bilinear",
                align_corners=False
            )                                            # [B,P,h,w]

        fused = torch.cat([img_emb, masks_ds], dim=1) # [B,256+P,h,w]

        formation_logits = self.formation_head(fused)

        return masks, formation_logits