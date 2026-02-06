import os
import sys
import urllib.request
import torch.nn as nn
from segment_anything import sam_model_registry

from .backbone.sam_backbone import SAMBackbone
from .prompt.sam_prompt_encoder import SAMPromptEncoder
from .decoder.sam_mask_decoder import SAMMaskDecoder

class SAMSegmenter(nn.Module):
    def __init__(self, sam_type="vit_h", ckpt_dir="./model/models", unfreeze_last_blocks=0):
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


        self.backbone = SAMBackbone(
            sam=sam,
            unfreeze_last_blocks=unfreeze_last_blocks
        )

        self.prompt = SAMPromptEncoder(sam)
        self.decoder = SAMMaskDecoder(sam)

    def forward(self, x):
        img_emb = self.backbone(x)
        sparse_emb, dense_emb = self.prompt()
        return self.decoder(
            image_embeddings=img_emb,
            sparse_emb=sparse_emb,
            dense_emb=dense_emb,
            out_size=x.shape[-2:],
        )