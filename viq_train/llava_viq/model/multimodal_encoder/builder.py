import os
from .encoders.siglip_vit_anyres_viq import AnyResViqWrapper
from .encoders.siglip_vit_anyres import SigLIPViTAnysizeWrapper

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if vision_tower in ["viq"]:
        return AnyResViqWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ['siglip_vit_anyres']:
        print("Buiding SigLIPViTAnyresWrapper...")
        return SigLIPViTAnysizeWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_vision_tower_two(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower_two', getattr(vision_tower_cfg, 'vision_tower_two', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    raise NotImplementedError(f"do not support vit two")
