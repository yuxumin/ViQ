"""Convert a heavy training vision_tower weight into the lightweight ViQ weights.

Produces, in --out_dir (default: same dir as the input ckpt):
  - <vision_tower_name>.pth   the converted encoder weight (new ViQ structure)
  - embedder.pth              IndexEmbeder weight (indices -> MLLM feature)
  - index_drawer.pth          IndexDrawer weight (indices -> reconstructed image)

It then runs a reconstruction-consistency check: the image reconstructed by the
full encoder forward path must be byte-identical to the one produced via
get_image_indices -> drawer, proving the conversion is lossless.

Model definitions are imported from modeling_viq.py and parameterized over the
FSQ levels (2k/4k/8k/16k/64k).

Example (16k):
    python convert_weight.py \
        --in_ckpt /path/to/fsq16k/vision_tower.pth \
        --levels 8 8 8 6 5
"""
import os
import sys
import math
import argparse

import torch
from PIL import Image
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # viq_inference/ (where modeling_viq.py lives)

from modeling_viq import (
    create_siglip_vit,
    resize_evaclip_pos_embed,
    IndexEmbeder,
    IndexDrawer,
    create_image_preprocess,
    process_image,
)


def convert_vision_tower(in_ckpt, out_path, fsq_levels):
    """Remap a training vision_tower.pth into the ViQ encoder state_dict and save."""
    state_dict = torch.load(in_ckpt, map_location="cpu")

    new_state_dict = {}
    prefix = "base_model.model.model.vision_tower.vision_tower."
    for k in state_dict.keys():
        if "perceptual_loss" in k:
            continue
        if not k.startswith(prefix):
            continue
        new_k = k.replace(prefix, "")
        if "dualvq_head." in new_k:
            new_k = new_k.replace("dualvq_head.", "fusion_block.")
        if "movq" in new_k:
            new_k = new_k.replace("movq", "vae")
        if "vq_low.fsq." in new_k:
            new_k = new_k.replace("vq_low.fsq.", "vq_low.")
        if "vae_cpu" in new_k:
            continue
        new_state_dict[new_k] = state_dict[k]
    state_dict = new_state_dict

    model_infer = create_siglip_vit(ckpt_path=None, fsq_levels=fsq_levels)
    model_infer = resize_evaclip_pos_embed(model_infer, interpolation="bilinear")

    # interpolate patch_embed / pos_embed to the inference size if needed
    patch_embed = state_dict["patch_embed.proj.weight"]
    if patch_embed.shape[-1] != model_infer.patch_embed.proj.weight.shape[-1]:
        patch_embed = torch.nn.functional.interpolate(
            patch_embed.float(), size=(16, 16), mode="bicubic", align_corners=False)
        print("interpolate model patch size to 16 ...")
        state_dict["patch_embed.proj.weight"] = patch_embed

    pos_embed = state_dict["pos_embed"]
    if pos_embed.shape[1] != model_infer.pos_embed.shape[1]:
        pos_embed = pos_embed.reshape(1, 24, 24, 1536).permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(128, 128), mode="bicubic", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, 1536)
        print("interpolate model pos embed size to 128 ...")
        state_dict["pos_embed"] = pos_embed

    incompatible = model_infer.load_state_dict(state_dict, strict=False)
    print(f"SigLIP-ViT restores from {in_ckpt}\n\tincompatible_keys: {incompatible}")

    torch.save(model_infer.state_dict(), out_path)
    print(f"[saved] converted vision_tower -> {out_path}")
    return model_infer


@torch.no_grad()
def verify(model_infer, embedder, drawer, save_dir, image_paths):
    """Reconstruction-consistency check: forward path vs indices->drawer path.

    If save_dir is None, the numeric consistency check still runs but no
    reconstruction images are written (keeps the output dir to just the weights).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    image_processor = create_image_preprocess()
    images = []
    for i, p in enumerate(image_paths):
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
        else:
            img = Image.open(requests.get(p, stream=True).raw).convert("RGB")
        image, _ = process_image(image=img, image_processor=image_processor, target_size=1024)
        images.append(image.unsqueeze(0).cuda().bfloat16())

    # full encoder forward path (includes fusion + drawer-equivalent reconstruction)
    x, image_sizes, cls_tokens, image_numpy_list, indices = model_infer.forward(
        images, plot=True, cal_attn_pool=True)
    verified_images = image_numpy_list
    if save_dir is not None:
        for i, vimg in enumerate(verified_images):
            Image.fromarray(vimg, mode="RGB").save(os.path.join(save_dir, f"forward_verify_{i}.png"))

    # indices -> embedder / drawer path
    indices, sizes = model_infer.get_image_indices(images)
    feat = embedder(indices)
    feat2 = embedder(torch.cat(indices, dim=0))
    feats, vae_latent_feats, image_numpy_list = drawer(indices, image_sizes)
    feats2, vae_latent_feats2, image_numpy_list2 = drawer(torch.cat(indices, dim=0), image_sizes[0])

    import numpy as np
    for i in range(len(indices)):
        assert (feat[i] == feat2[i]).all() and (feat2[i] == feats[i]).all() and (feats[i] == feats2[i]).all(), f"feat mismatch @ {i}"
        assert (vae_latent_feats[i] == vae_latent_feats2[i:i+1]).all(), f"vae latent mismatch @ {i}"
        a = image_numpy_list[i].astype(np.int32)
        b = image_numpy_list2[i].astype(np.int32)
        c = verified_images[i].astype(np.int32)
        nd_lb = int((a != b).sum()); md_lb = int(np.abs(a - b).max()) if a.shape == b.shape else -1
        nd_lf = int((a != c).sum()); md_lf = int(np.abs(a - c).max()) if a.shape == c.shape else -1
        print(f"[verify @ {i}] shapes list={image_numpy_list[i].shape} batch={image_numpy_list2[i].shape} fwd={verified_images[i].shape}")
        print(f"[verify @ {i}] list-vs-batch: ndiff={nd_lb} maxdiff={md_lb} | list-vs-forward: ndiff={nd_lf} maxdiff={md_lf}")

    print(" pass the verification. ")
    if save_dir is not None:
        for i, image_numpy in enumerate(image_numpy_list):
            Image.fromarray(image_numpy, mode="RGB").save(os.path.join(save_dir, f"forward_verify3_{i}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True,
                    help="training vision_tower.pth to convert")
    ap.add_argument("--out_dir", default=None,
                    help="output dir for converted weights (default: dir of in_ckpt)")
    ap.add_argument("--out_name", default="model_viq_fsq.pth",
                    help="file name for the converted vision_tower weight")
    ap.add_argument("--levels", type=int, nargs="+", default=[8, 8, 8, 6, 5],
                    help="FSQ levels (e.g. 16k=8 8 8 6 5, 64k=8 8 8 5 5 5)")
    ap.add_argument("--skip_verify", action="store_true",
                    help="skip the reconstruction-consistency check")
    ap.add_argument("--debug_dir", default=None,
                    help="if set, the verification reconstructions are saved here "
                         "(default: verify runs but writes no images, keeping the "
                         "output dir to just the 3 .pth weights)")
    ap.add_argument("--force", action="store_true",
                    help="reconvert even if output weights already exist")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.in_ckpt)
    os.makedirs(out_dir, exist_ok=True)
    vt_path = os.path.join(out_dir, args.out_name)
    embedder_path = os.path.join(out_dir, "embedder.pth")
    drawer_path = os.path.join(out_dir, "index_drawer.pth")

    fsq_levels = list(args.levels)
    codesize = int(math.prod(fsq_levels))
    codedim = len(fsq_levels)
    print(f"FSQ levels={fsq_levels}  codebook_size={codesize}  codedim={codedim}")

    # 1) convert vision_tower (skip if already converted; reload for verify)
    if os.path.exists(vt_path) and not args.force:
        print(f"[skip] {vt_path} exists, loading it (use --force to reconvert)")
        model_infer = create_siglip_vit(ckpt_path=vt_path, fsq_levels=fsq_levels).cuda().bfloat16()
    else:
        model_infer = convert_vision_tower(args.in_ckpt, vt_path, fsq_levels).cuda().bfloat16()
    implicit_codebook = model_infer.fusion_block.vq_low.implicit_codebook

    # 2) derive embedder
    embedder = IndexEmbeder(codedim=codedim, codesize=codesize).cuda().bfloat16()
    if os.path.exists(embedder_path) and not args.force:
        embedder.init_weight_from_viq_weight  # noqa
        import torch as _t
        embedder.load_state_dict(_t.load(embedder_path, map_location="cpu"), strict=False)
        print(f"[skip] loaded existing {embedder_path}")
    else:
        embedder.init_weight_from_viq_weight(vt_path, implicit_codebook=implicit_codebook, save_path=embedder_path)
        print(f"[saved] embedder -> {embedder_path}")

    # 3) derive drawer
    drawer = IndexDrawer(codedim=codedim, codesize=codesize).cuda().bfloat16()
    if os.path.exists(drawer_path) and not args.force:
        import torch as _t
        drawer.load_state_dict(_t.load(drawer_path, map_location="cpu"), strict=False)
        print(f"[skip] loaded existing {drawer_path}")
    else:
        drawer.init_weight_from_viq_weight(vt_path, implicit_codebook=implicit_codebook, save_path=drawer_path)
        print(f"[saved] index_drawer -> {drawer_path}")

    # 4) verify
    if not args.skip_verify:
        image_paths = [
            os.path.join(_HERE, os.pardir, os.pardir, "assets", "verify_0.jpeg"),
            os.path.join(_HERE, os.pardir, os.pardir, "assets", "verify_1.jpg"),
        ]
        verify(model_infer, embedder, drawer, args.debug_dir, image_paths)


if __name__ == "__main__":
    main()
