#!/bin/bash

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

# Optional: set http(s)_proxy if your machine needs a proxy. Leave unset otherwise.
# export http_proxy=...
# export https_proxy=...

pip show deepspeed
which python

pkill torchrun

# Resolve repo root from this script's own location (depth-independent).
# This script lives at <repo>/scripts/example.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/viq_train" ]; do REPO_ROOT="$(dirname "$REPO_ROOT")"; done
export VIQ_ROOT="$REPO_ROOT/viq_train"

MASTER_ADDR=${CHIEF_IP:-127.0.0.1}
nnode=1
nrank=$((INDEX%nnode))

MODEL_NAME=Qwen2.5-0.5B
mkdir -p /home/checkpoints/
mkdir -p /home/data/
mkdir -p /home/models/

# Stage source weights into the local /home/models cache.
# Override these roots if your weights live elsewhere.
#   MODELS_ROOT  : HF source models (Qwen2.5-0.5B, siglip2_g_384_16, Qwen-Image)
#   WEIGHTS_ROOT : downloaded ViQ weights (anyres vit checkpoints)
MODELS_ROOT="${MODELS_ROOT:-$REPO_ROOT}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$REPO_ROOT/../ViQ_weights}"

rsync -ah --progress $MODELS_ROOT/siglip2_g_384_16/open_clip_pytorch_model.bin /home/models/
rsync -ah --progress $MODELS_ROOT/$MODEL_NAME /home/models/
rsync -ah --progress $WEIGHTS_ROOT/anyres_vit/giant1b/siglip2_g_anyres_s4.pth /home/models/
rsync -ah --progress --exclude text_encoder* --exclude transformer  $MODELS_ROOT/Qwen-Image /home/models/


# Runtime/launch knob (not a ViQ setting): the PyTorch CUDA allocator. Reduces
# fragmentation OOMs with variable-resolution inputs; tune for your hardware.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
###############################################################################
#                          Settings (fixed for ViQ)                           #
#  These rarely change when training ViQ -- leave them as-is unless you know   #
#  you need otherwise.                                                         #
#                                                                             #
#  Each tunable below is its own block: a "# ===== NAME =====" rule, a short   #
#  description, then the `export`. Scan for the rules to find a variable.      #
###############################################################################
echo $MASTER_ADDR; echo $nnode; echo $nrank
export PYTHONPATH=$VIQ_ROOT:$PYTHONPATH

# (VIT_WITH_GRAD -- train the vision tower -- and VQ_GEN_VIT -- build the
#  add quantizer and reconstruction path for vit -- are always on for ViQ and are no longer switches.)

# ===== EXP_NAME / THIS_EXP_NAME =============================================
# Names for this run. Change them per run so outputs don't clash.
#   EXP_NAME       -> the trainer's --run_name and the checkpoint/output dir.
#   THIS_EXP_NAME  -> subfolder for saved training plots (reconstruction /
#                     debug visualizations), at:
#                     viq_train/llava_viq/training_plot/<THIS_EXP_NAME>/
EXP_NAME="test"
export THIS_EXP_NAME='test'

# ===== QUIET_PARAM_LOG  (optional; off here) ================================
# Suppress the long per-parameter freeze/unfreeze + "Trainable Param" dump
# printed at model setup. Set it (uncomment) for clean logs; the final
# trainable-parameter total is still printed.
export QUIET_PARAM_LOG=1

###############################################################################
#                          Quantizer (the VQ head)                            #
###############################################################################
# ViQ has a dual-branch head: a "low" branch (the codes you actually use for
# understanding / reconstruction) and a "high" branch (auxiliary, usually faked
# out in the released recipe). Each branch picks a quantizer TYPE, a codebook
# SIZE, and a feature-space LIMIT.

# ===== FSQ<N>K  (codebook preset) ===========================================
# When VQ_LOW_TYPE='fsq', the per-level grid is chosen by exactly ONE of these
# flags (set one, leave the rest unset). They map to the README sizes:
#     FSQ2K    -> levels [8,8,4,3,3]    (codebook 2304)
#     FSQ4K    -> levels [8,8,4,4,4]    (codebook 4096)
#     FSQ8K    -> levels [8,8,8,4,4]    (codebook 8192)
#     FSQ16K   -> levels [8,8,8,6,5]    (codebook 15360)   <-- this example
#     (none)   -> levels [8,8,8,5,5,5]  (codebook 64000, the "64k" model)
#     FSQ16KV2 -> levels [2]*14         (binary-style, experimental)
export FSQ16K=1

# ===== ADD_PRE_ATTN =========================================================
# Number of self-attention blocks BEFORE quantization that expand each patch
# into 2x2 codes (0 = disabled). The released recipe uses 3.
export ADD_PRE_ATTN=3

# ===== ADD_POST_CAUSAL_ATTN =================================================
# Number of causal attention blocks AFTER quantization (0 = disabled).
export ADD_POST_CAUSAL_ATTN=0

# ===== ENABLE_ROPE ==========================================================
# Add 2D rotary position embedding inside the FSQ head so codes encode spatial
# resolution. Requires ADD_PRE_ATTN>0 or ADD_POST_CAUSAL_ATTN>0.
export ENABLE_ROPE=1

# ===== VQ_LOW_TYPE  (low branch = the main codes) ===========================
# Quantizer family. Options:
#   'fsq'                          Finite Scalar Quantization (recommended; this example)
#   'simvq' / 'simvq-normal' /
#   'simvq-middle' / 'simvq-wide'  SimVQ with codebook_dim 32 / 256 / 128 / 512
#   'packer'                       VQ packer
#   'fake' / 'fake-normal' /
#   'fake-middle' / 'fake-wide'    pass-through (no real quantization; for ablations)
export VQ_LOW_TYPE='fsq'

# ===== VQ_LOW_SIZE ==========================================================
# Codebook size for codebook-based quantizers (simvq/packer/fake).
# Ignored by 'fsq' (FSQ derives its size from the FSQ* preset above).
export VQ_LOW_SIZE=$((2 ** 15))

# ===== VQ_LOW_LIMIT =========================================================
# How features are constrained before quantization. A feature-space "proximal"
# projection that typically REPLACES a real quantizer (used in stage 2-1 with a
# 'fake' VQ, where there is no actual codebook yet); not tied to any quantizer.
# Options:
#   'none'        no constraint
#   'tanh'        squash to (-1, 1)
#   'l2'          L2-normalize (project onto the unit sphere)
#   'l_infinite'  L-infinity norm: project onto the unit-hypercube surface
#                 (||f||_inf == 1) -- the ViQ proximal representation (stage 2-1)
#   'escape'      bypass the limit (only valid for fake/packer; NOT for simvq)
export VQ_LOW_LIMIT='none'

# ----------------------------------------------------------------------------
# High branch: auxiliary, faked out in the released recipe. VQ_HIGH_* use the
# same option sets as their VQ_LOW_* counterparts above. The released recipe
# disables it with a 'fake' quantizer that 'escape's straight through, so only
# the low-branch codes matter.
# ----------------------------------------------------------------------------

# ===== VQ_HIGH_TYPE =========================================================
# High-branch quantizer family (same options as VQ_LOW_TYPE).
export VQ_HIGH_TYPE='fake'

# ===== VQ_HIGH_SIZE =========================================================
# High-branch codebook size (same meaning as VQ_LOW_SIZE).
export VQ_HIGH_SIZE=$((2 ** 15))

# ===== VQ_HIGH_LIMIT ========================================================
# High-branch feature constraint (same options as VQ_LOW_LIMIT).
export VQ_HIGH_LIMIT='escape'

###############################################################################
#                              Design / tricks                                #
###############################################################################
# (SYMMETRY_VQ -- symmetric encode/decode quantization -- is always on for ViQ
#  and is no longer a switch.)

# ===== VQ_LOW_ENABLE_TOKEN_SHUFFLE ==========================================
# Shuffle tokens before low-branch quantization (regularization for the
# any-resolution packing).
export VQ_LOW_ENABLE_TOKEN_SHUFFLE=1

# ===== VQ_LOW_ENABLE_LAYERNORM_TRICK ========================================
# Apply the no-affine LayerNorm pre/post-quantize trick on the low branch
# (stabilizes the quantized feature space).
export VQ_LOW_ENABLE_LAYERNORM_TRICK=1

# High-branch counterparts (only relevant if you actually train the high branch):
# export VQ_HIGH_ENABLE_TOKEN_SHUFFLE=1
# export VQ_HIGH_ENABLE_LAYERNORM_TRICK=1

###############################################################################
#                                  Loss                                       #
###############################################################################

# ===== RETURN_FEAT_REC_LOSS  (optional; off here) ===========================
# Add a feature-reconstruction loss on the quantized features.
# export RETURN_FEAT_REC_LOSS=1

# ===== TRAIN_CLS_TOKEN ======================================================
# Enable the self-distillation loss on the semantic/CLS token (aligns the
# any-resolution student to the fixed-resolution teacher).
export TRAIN_CLS_TOKEN=1

# ===== CLS_DISTILL_FEATURE_TYPE =============================================
# Which branch's feature the CLS distillation targets. Options: 'low' | 'high'.
export CLS_DISTILL_FEATURE_TYPE='low'

###############################################################################
#                       MoVQ / VAE reconstruction decoder                     #
###############################################################################

# ===== ENABLE_MOVQ_DECODER ==================================================
# Attach a pixel decoder so codes can be decoded back to images (the
# reconstruction path). Comment this whole MoVQ block out for an
# understanding-only run that does not need the VAE.
export ENABLE_MOVQ_DECODER=1

# ===== MOVQ_TYPE ============================================================
# Decoder backend. Options:
#   'qwen_vae'                     frozen Qwen-Image VAE (this example; needs VAE_PATH)
#   'qwen_vae_trainable'           same but trainable
#   'fixed_vae' / 'fixed_vae_f16'  other fixed VAEs (need VAE_PATH)
#   'movq'                         built-in MoVQ decoder (no external VAE)
#   'vqvit'                        ViT-based decoder
export MOVQ_TYPE='qwen_vae'

# ===== VAE_PATH =============================================================
# Path to the VAE weights (required for the *_vae MOVQ_TYPE values).
export VAE_PATH='/home/models/Qwen-Image/vae'

# ===== MOVQ_PREPROCESS_EMBED_DIM ============================================
# Channel width of the adapter feeding the decoder (the working dim used by
# MOVQ_PREPROCESS_TYPE below).
export MOVQ_PREPROCESS_EMBED_DIM=1536

# ===== MOVQ_PREPROCESS_TYPE =================================================
# The adapter that maps encoder features into the decoder. All variants first
# project the incoming feature to MOVQ_PREPROCESS_EMBED_DIM. Attention blocks
# use the encoder's num_heads (12) and mlp_ratio (4.0). Options:
#   'none'          Identity (feed features straight to the decoder)
#   'mlp'           a single MLP  (-> MOVQ_PREPROCESS_EMBED_DIM)
#   'attn'          MLP + LayerNorm + 3 Transformer blocks  (this example)
#   'attn-shallow'  MLP + LayerNorm + 2 Transformer blocks  (lighter)
export MOVQ_PREPROCESS_TYPE='attn'

# ===== MOVQ_PLUGIN_POSITION =================================================
# Where the decoder taps the encoder. Options:
#   'after_postprocess' (this example) | 'oryx' | 'after_sa' | 'after_ts' |
#   'after_quant_8x' | 'after_quant_16x' | 'after_concat'.
export MOVQ_PLUGIN_POSITION='after_postprocess'

###############################################################################
#                          Architecture (head shape)                          #
###############################################################################

# ===== VQ_LOW_PREPROCESS_TYPE ===============================================
# Low-branch block BEFORE quantization. Options:
#   'none' | 'attn' | 'attn-large' | 'attn-large-shallow'.
export VQ_LOW_PREPROCESS_TYPE='attn'

# ===== VQ_HIGH_PREPROCESS_TYPE ==============================================
# High-branch pre-quant block. Options: 'none' | 'attn'.
export VQ_HIGH_PREPROCESS_TYPE='none'

# ===== VQ_LOW_POSTPROCESS_TYPE ==============================================
# Low-branch block AFTER quantization. Options:
#   'none' | 'mlp' | 'attn' | 'attn-large' | 'attn-deep' |
#   'casual_attn' | 'casual_attn-wide' | 'resualmlp' | 'resualmlp-light'.
export VQ_LOW_POSTPROCESS_TYPE='resualmlp-light'

# ===== VQ_HIGH_POSTPROCESS_TYPE =============================================
# High-branch post-quant block. Options: 'none' | 'mlp' | 'attn' | 'casual_attn'.
export VQ_HIGH_POSTPROCESS_TYPE='none'

# ===== MLLM_FEATURE_TYPE ====================================================
# Which feature is handed to the LLM. Options:
#   'low' (the ViQ codes; this example) | 'high' | 'concat'.
export MLLM_FEATURE_TYPE='low'

SAVEROOT="$REPO_ROOT/temp"

cd "$REPO_ROOT"
echo "Default path: $SAVEROOT"

TRAIN_SCRIPT="llava_viq/train/train.py"

torchrun  --nproc_per_node 1 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=24824 \
    "$VIQ_ROOT/$TRAIN_SCRIPT" \
    --deepspeed "$REPO_ROOT/scripts/zero1.json" \
    --run_name $EXP_NAME \
    --lora_enable True \
    --model_name_or_path /home/models/$MODEL_NAME  \
    --sample_independently True \
    --version v1_qwen2 \
    --data_path "$REPO_ROOT/scripts/example_dataset/example.json" \
    --vision_tower viq  \
    --unfreeze_mm_vision_tower True \
    --mm_projector_type oryx_mlp_v2_twoview \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SAVEROOT/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --min_lr_ratio 0.1 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard



