"""ViQ inference using the converted weights (produced by ./converter/convert_weight.py).

Loads the per-size converted weights under ./converter/converted_<size>/ and
reuses the model definitions from ./modeling_viq.py.

Data flow:  image -> discrete codes (indices) -> { MLLM feature | reconstructed image }

CLI:
    # run one size on demo / local images (weights from ./converter by default)
    python ViQ.py --size 16k
    # all sizes
    python ViQ.py --size all
    # your own images
    python ViQ.py --size 16k --images a.jpg b.png
    # load weights from an external root (or set $VIQ_WEIGHTS_ROOT)
    python ViQ.py --size 16k --weights_root /mnt/castle/castle/ViQ_weights/ViQ

Programmatic:
    from ViQ import load_viq
    vq = load_viq('16k')                                       # default root
    vq = load_viq('16k', '/mnt/castle/castle/ViQ_weights/ViQ')  # external root
    indices, sizes = vq.forward_indices(images)       # encode
    feats = vq.embedder(indices)                      # -> LLM features
    _, vae_latent, recon_np = vq.drawer(indices, sizes)  # -> reconstructed images
"""
import os
import sys
import argparse

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from modeling_viq import AnyResViqVQWrapper, create_image_preprocess, process_image  # noqa: E402


# Default root holding the per-size converted weights (converted_<size>/...).
# Override with --weights_root or the VIQ_WEIGHTS_ROOT env var to load weights
# from anywhere, e.g. /mnt/castle/castle/ViQ_weights/ViQ.
DEFAULT_WEIGHTS_ROOT = os.environ.get(
    "VIQ_WEIGHTS_ROOT", os.path.join(_HERE, "converter"))


# size -> (levels, codebook_size). weight files live under <root>/converted_<size>/.
VIQ_SPECS = {
    "2k":  ([8, 8, 4, 3, 3],    2304),
    "4k":  ([8, 8, 4, 4, 4],    4096),
    "8k":  ([8, 8, 8, 4, 4],    8192),
    "16k": ([8, 8, 8, 6, 5],    15360),
    "64k": ([8, 8, 8, 5, 5, 5], 64000),
}


def _weight_root(size, weights_root=DEFAULT_WEIGHTS_ROOT):
    return os.path.join(weights_root, f"converted_{size}", f"model_viq_fsq_{size}.pth")


def load_viq(size, weights_root=DEFAULT_WEIGHTS_ROOT):
    """Build an AnyResViqVQWrapper for the given ViQ size, on cuda/bf16.

    weights_root: directory containing converted_<size>/ subfolders. Each holds
    model_viq_fsq_<size>.pth plus the embedder.pth / index_drawer.pth that the
    wrapper auto-discovers alongside it.
    """
    if size not in VIQ_SPECS:
        raise ValueError(f"unknown size {size!r}; choose from {list(VIQ_SPECS)}")
    levels, codebook_size = VIQ_SPECS[size]
    config = {
        "weight_root": _weight_root(size, weights_root),  # embedder.pth / index_drawer.pth auto-found alongside
        "codebook_size": codebook_size,
        "levels": levels,
    }
    return AnyResViqVQWrapper(config=config).cuda().bfloat16()


def _load_images(image_processor, image_paths, target_size=1024):
    import requests
    images = []
    for p in image_paths:
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
        else:
            img = Image.open(requests.get(p, stream=True).raw).convert("RGB")
        image, _size = process_image(image=img, image_processor=image_processor, target_size=target_size)
        images.append(image.unsqueeze(0).cuda().bfloat16())
    return images


@torch.no_grad()
def run_size(size, image_paths, out_root, target_size=1024, weights_root=DEFAULT_WEIGHTS_ROOT):
    print("\n" + "=" * 70)
    print(f"ViQ-{size}  weight={_weight_root(size, weights_root)}")
    print("=" * 70)
    vq = load_viq(size, weights_root)
    images = _load_images(vq.image_processor, image_paths, target_size)

    # encode: image -> discrete codes
    indices, image_sizes = vq.forward_indices(images)
    # codes -> LLM feature
    feats = vq.embedder(indices)
    # codes -> reconstructed image
    _feats, vae_latent_feats, image_numpy_list = vq.drawer(indices, image_sizes)

    # ===== unified diagnostic fingerprint (compare against ViQ.py) =====
    print(f"##### DBG [ViQ-{size}] #####")
    for i in range(len(indices)):
        print(f"DBG indices[{i}] shape={tuple(indices[i].shape)} sum={indices[i].float().sum().item()}")
        print(f"DBG embed_feat[{i}] shape={tuple(feats[i].shape)} absum={feats[i].float().abs().sum().item():.4f}")
        print(f"DBG vae_latent[{i}] shape={tuple(vae_latent_feats[i].shape)} absum={vae_latent_feats[i].float().abs().sum().item():.4f}")
        print(f"DBG recon[{i}] shape={image_numpy_list[i].shape} sum={int(image_numpy_list[i].astype('int64').sum())}")
    # ===================================================================

    save_dir = os.path.join(out_root, f"viq_new_{size}")
    os.makedirs(save_dir, exist_ok=True)
    for i, image_numpy in enumerate(image_numpy_list):
        sp = os.path.join(save_dir, f"recon_{i}.png")
        Image.fromarray(image_numpy, mode="RGB").save(sp)

    del vq
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="16k", help="2k/4k/8k/16k/64k or 'all'")
    ap.add_argument("--weights_root", default=DEFAULT_WEIGHTS_ROOT,
                    help="dir holding converted_<size>/ subfolders "
                         "(default: ./converter, or $VIQ_WEIGHTS_ROOT; "
                         "e.g. /mnt/castle/castle/ViQ_weights/ViQ)")
    ap.add_argument("--images", nargs="*", default=None,
                    help="local image paths; default uses the two assets/verify images")
    ap.add_argument("--target_size", type=int, default=1024)
    ap.add_argument("--out_dir", default=None, help="default: ./viq_new_out")
    args = ap.parse_args()

    image_paths = args.images or [
        os.path.join(_HERE, os.pardir, "assets", "verify_0.jpeg"),
        os.path.join(_HERE, os.pardir, "assets", "verify_1.jpg"),
    ]
    out_root = args.out_dir or os.path.join(_HERE, "viq_new_out")

    sizes = list(VIQ_SPECS) if args.size == "all" else [args.size]
    for s in sizes:
        run_size(s, image_paths, out_root, args.target_size, args.weights_root)


if __name__ == "__main__":
    main()
