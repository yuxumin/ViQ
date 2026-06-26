import os

FSQ16K = 'FSQ16K' in os.environ
FSQ16KV2 = 'FSQ16KV2' in os.environ
FSQ8K = 'FSQ8K' in os.environ
FSQ4K = 'FSQ4K' in os.environ
FSQ2K = 'FSQ2K' in os.environ

# VIT_WITH_GRAD is always enabled for ViQ (the vision tower is trained, not frozen).
VIT_WITH_GRAD = True

if 'CLS_WITH_ORIGIN_LAYER_OUTPUT' in os.environ:
    CLS_WITH_ORIGIN_LAYER_OUTPUT = True
    print("CLS_WITH_ORIGIN_LAYER_OUTPUT is set")
else:
    CLS_WITH_ORIGIN_LAYER_OUTPUT = False


if 'VQ_LOW_TYPE' in os.environ:
    VQ_LOW_TYPE = os.environ['VQ_LOW_TYPE']
else:
    VQ_LOW_TYPE = 'simvq'

if 'VQ_LOW_SIZE' in os.environ:
    VQ_LOW_SIZE = int(os.environ['VQ_LOW_SIZE'])
else:
    VQ_LOW_SIZE = 2**15

if 'VQ_LOW_LIMIT' in os.environ:
    VQ_LOW_LIMIT = os.environ['VQ_LOW_LIMIT']
else:
    VQ_LOW_LIMIT = "none"

if 'VQ_HIGH_TYPE' in os.environ:
    VQ_HIGH_TYPE = os.environ['VQ_HIGH_TYPE']
else:
    VQ_HIGH_TYPE = 'simvq'

if 'VQ_HIGH_SIZE' in os.environ:
    VQ_HIGH_SIZE = int(os.environ['VQ_HIGH_SIZE'])
else:
    VQ_HIGH_SIZE = 2**15

if 'VQ_HIGH_LIMIT' in os.environ:
    VQ_HIGH_LIMIT = os.environ['VQ_HIGH_LIMIT']
else:
    VQ_HIGH_LIMIT = "none"

if 'VQ_HIGH_ENABLE_TOKEN_SHUFFLE' in os.environ:
    VQ_HIGH_ENABLE_TOKEN_SHUFFLE = True
else:
    VQ_HIGH_ENABLE_TOKEN_SHUFFLE = False

if 'VQ_HIGH_ENABLE_LAYERNORM_TRICK' in os.environ:
    VQ_HIGH_ENABLE_LAYERNORM_TRICK = True
else:
    VQ_HIGH_ENABLE_LAYERNORM_TRICK = False

if 'VQ_LOW_ENABLE_TOKEN_SHUFFLE' in os.environ:
    VQ_LOW_ENABLE_TOKEN_SHUFFLE = True
else:
    VQ_LOW_ENABLE_TOKEN_SHUFFLE = False

if 'VQ_LOW_ENABLE_LAYERNORM_TRICK' in os.environ:
    VQ_LOW_ENABLE_LAYERNORM_TRICK = True
else:
    VQ_LOW_ENABLE_LAYERNORM_TRICK = False

if 'ENABLE_MOVQ_DECODER' in os.environ:
    ENABLE_MOVQ_DECODER = True
else:
    ENABLE_MOVQ_DECODER = False

if 'MOVQ_TYPE' in os.environ:
    MOVQ_TYPE = os.environ['MOVQ_TYPE']
    assert MOVQ_TYPE in ['movq', 'vqvit', 'fixed_vae', 'fixed_vae_f16', 'qwen_vae', 'qwen_vae_trainable']
else:
    MOVQ_TYPE = 'movq'

if MOVQ_TYPE == 'fixed_vae' or MOVQ_TYPE == 'qwen_vae' or MOVQ_TYPE == 'qwen_vae_trainable':
    assert 'VAE_PATH' in os.environ
    VAE_PATH = os.environ['VAE_PATH']
else:
    VAE_PATH = None

if "VIQ_SO400M" in os.environ:
    VIQ_SO400M = True
else:
    VIQ_SO400M = False

if "USE_FIRST_CODE_FOR_REC" in os.environ:
    USE_FIRST_CODE_FOR_REC = True
else:
    USE_FIRST_CODE_FOR_REC = False

if 'THIS_EXP_NAME' in os.environ:
    THIS_EXP_NAME = os.environ['THIS_EXP_NAME']
else:
    THIS_EXP_NAME = 'unknown'

if 'MOVQ_PREPROCESS_EMBED_DIM' in os.environ:
    MOVQ_PREPROCESS_EMBED_DIM = int(os.environ['MOVQ_PREPROCESS_EMBED_DIM'])
else:
    MOVQ_PREPROCESS_EMBED_DIM = 64

if 'MOVQ_PREPROCESS_TYPE' in os.environ:
    MOVQ_PREPROCESS_TYPE = os.environ['MOVQ_PREPROCESS_TYPE']
else:
    MOVQ_PREPROCESS_TYPE = 'attn'

if 'MOVQ_PLUGIN_POSITION' in os.environ:
    MOVQ_PLUGIN_POSITION = os.environ['MOVQ_PLUGIN_POSITION']
else:
    MOVQ_PLUGIN_POSITION = 'oryx'

if 'MOVQ_DISABLE_PERCEPTUAL_LOSS' in os.environ:
    MOVQ_DISABLE_PERCEPTUAL_LOSS = True
else:
    MOVQ_DISABLE_PERCEPTUAL_LOSS = False

if 'RETURN_FEAT_REC_LOSS' in os.environ:
    RETURN_FEAT_REC_LOSS = True
else:
    RETURN_FEAT_REC_LOSS = False

if 'TRAIN_CLS_TOKEN' in os.environ:
    TRAIN_CLS_TOKEN = True
else:
    TRAIN_CLS_TOKEN = False

if 'VQ_LOW_PREPROCESS_TYPE' in os.environ:
    VQ_LOW_PREPROCESS_TYPE = os.environ['VQ_LOW_PREPROCESS_TYPE']
else:
    VQ_LOW_PREPROCESS_TYPE = 'attn'

if 'VQ_HIGH_PREPROCESS_TYPE' in os.environ:
    VQ_HIGH_PREPROCESS_TYPE = os.environ['VQ_HIGH_PREPROCESS_TYPE']
else:
    VQ_HIGH_PREPROCESS_TYPE = 'attn'

if 'VQ_LOW_POSTPROCESS_TYPE' in os.environ:
    VQ_LOW_POSTPROCESS_TYPE = os.environ['VQ_LOW_POSTPROCESS_TYPE']
else:
    VQ_LOW_POSTPROCESS_TYPE = 'attn'

if 'VQ_HIGH_POSTPROCESS_TYPE' in os.environ:
    VQ_HIGH_POSTPROCESS_TYPE = os.environ['VQ_HIGH_POSTPROCESS_TYPE']
else:
    VQ_HIGH_POSTPROCESS_TYPE = 'attn'

if 'CLS_DISTILL_FEATURE_TYPE' in os.environ:
    CLS_DISTILL_FEATURE_TYPE = os.environ['CLS_DISTILL_FEATURE_TYPE']
    assert CLS_DISTILL_FEATURE_TYPE in ['high', 'low'], f"CLS_DISTILL_FEATURE_TYPE should be 'high' or 'low', but got {CLS_DISTILL_FEATURE_TYPE}"
else:
    CLS_DISTILL_FEATURE_TYPE = 'high'

if 'MLLM_FEATURE_TYPE' in os.environ:
    MLLM_FEATURE_TYPE = os.environ['MLLM_FEATURE_TYPE']
    assert MLLM_FEATURE_TYPE in ['high', 'concat', 'low'], f"MLLM_FEATURE_TYPE should be 'high' or 'concat', but got {MLLM_FEATURE_TYPE}"
else:
    MLLM_FEATURE_TYPE = 'high'

if 'SKIP_RECON_VAE' in os.environ:
    SKIP_RECON_VAE = True
else:
    SKIP_RECON_VAE = False


# SYMMETRY_VQ is always enabled for ViQ (symmetric encode/decode quantization).
SYMMETRY_VQ = True

if 'ADD_AUX_LAYERS' in os.environ:
    assert SYMMETRY_VQ or VQ_LOW_LIMIT == 'escape'
    ADD_AUX_LAYERS = True
else:
    ADD_AUX_LAYERS = False

if 'EXTRA_SA_BEFORE_MAP' in os.environ:
    EXTRA_SA_BEFORE_MAP = True
else:
    EXTRA_SA_BEFORE_MAP = False

if 'USE_CPU_VIS' in os.environ:
    USE_CPU_VIS = True
else:
    USE_CPU_VIS = False

if 'LOWVRAM_MODE' in os.environ:
    LOWVRAM_MODE = True
else:
    LOWVRAM_MODE = False


if 'INFERENCE_IMAGE_SIZE' in os.environ:
    INFERENCE_IMAGE_SIZE = int(os.environ['INFERENCE_IMAGE_SIZE'])
else:
    INFERENCE_IMAGE_SIZE = 3

if 'RETURN_FSQ_INDEX' in os.environ:
    RETURN_FSQ_INDEX = True
else:
    RETURN_FSQ_INDEX = False

if 'USE_STAGE1_AS_TEACHER' in os.environ:
    USE_STAGE1_AS_TEACHER = True
    assert 'TEACHER_CKPT' in os.environ, "TEACHER_CKPT must be set when USE_STAGE1_AS_TEACHER is True"
    assert 'TEACHER_VQ_LOW_TYPE' in os.environ
    assert 'TEACHER_VQ_LOW_SIZE' in os.environ
    assert 'TEACHER_VQ_LOW_LIMIT' in os.environ
    assert 'TEACHER_VQ_HIGH_TYPE' in os.environ         
    assert 'TEACHER_VQ_HIGH_SIZE' in os.environ
    assert 'TEACHER_VQ_HIGH_LIMIT' in os.environ

    assert 'TEACHER_VQ_LOW_PREPROCESS_TYPE' in os.environ
    assert 'TEACHER_VQ_HIGH_PREPROCESS_TYPE' in os.environ
    assert 'TEACHER_VQ_LOW_POSTPROCESS_TYPE' in os.environ
    assert 'TEACHER_VQ_HIGH_POSTPROCESS_TYPE' in os.environ
    assert 'TEACHER_MLLM_FEATURE_TYPE' in os.environ
    assert 'CLS_DISTILL_FEATURE_TYPE' in os.environ

    TEACHER_CKPT = os.environ['TEACHER_CKPT']

    TEACHER_VQ_LOW_TYPE = os.environ['TEACHER_VQ_LOW_TYPE']
    TEACHER_VQ_LOW_SIZE = int(os.environ['TEACHER_VQ_LOW_SIZE'])
    TEACHER_VQ_LOW_LIMIT = os.environ['TEACHER_VQ_LOW_LIMIT']
    TEACHER_VQ_HIGH_TYPE = os.environ['TEACHER_VQ_HIGH_TYPE']
    TEACHER_VQ_HIGH_SIZE = int(os.environ['TEACHER_VQ_HIGH_SIZE'])
    TEACHER_VQ_HIGH_LIMIT = os.environ['TEACHER_VQ_HIGH_LIMIT']

    TEACHER_VQ_LOW_ENABLE_TOKEN_SHUFFLE = 'TEACHER_VQ_LOW_ENABLE_TOKEN_SHUFFLE' in os.environ
    TEACHER_VQ_LOW_ENABLE_LAYERNORM_TRICK = 'TEACHER_VQ_LOW_ENABLE_LAYERNORM_TRICK' in os.environ
    TEACHER_VQ_HIGH_ENABLE_TOKEN_SHUFFLE = 'TEACHER_VQ_HIGH_ENABLE_TOKEN_SHUFFLE' in os.environ
    TEACHER_VQ_HIGH_ENABLE_LAYERNORM_TRICK = 'TEACHER_VQ_HIGH_ENABLE_LAYERNORM_TRICK' in os.environ

    # TEACHER_VQ_LOW_ESCAPE_SHARED_PREPROCESS = os.environ.get('TEACHER_VQ_LOW_ESCAPE_SHARED_PREPROCESS', 'false').lower() != 'false'
    TEACHER_ENABLE_MOVQ_DECODER = False
    TEACHER_RETURN_FEAT_REC_LOSS = False
    TEACHER_RETURN_CLS_DISTILL_LOSS = True

    # TEACHER_MOVQ_PREPROCESS_EMBED_DIM = int(os.environ['TEACHER_MOVQ_PREPROCESS_EMBED_DIM'])
    # TEACHER_MOVQ_PREPROCESS_TYPE = os.environ['TEACHER_MOVQ_PREPROCESS_TYPE']
    # TEACHER_MOVQ_PLUGIN_POSITION = os.environ['TEACHER_MOVQ_PLUGIN_POSITION']
    TEACHER_VQ_LOW_PREPROCESS_TYPE = os.environ['TEACHER_VQ_LOW_PREPROCESS_TYPE']
    TEACHER_VQ_HIGH_PREPROCESS_TYPE = os.environ['TEACHER_VQ_HIGH_PREPROCESS_TYPE']
    TEACHER_VQ_LOW_POSTPROCESS_TYPE = os.environ['TEACHER_VQ_LOW_POSTPROCESS_TYPE']
    TEACHER_VQ_HIGH_POSTPROCESS_TYPE = os.environ['TEACHER_VQ_HIGH_POSTPROCESS_TYPE']
    TEACHER_MLLM_FEATURE_TYPE = os.environ['TEACHER_MLLM_FEATURE_TYPE']
    TEACHER_CLS_DISTILL_FEATURE_TYPE = os.environ['TEACHER_CLS_DISTILL_FEATURE_TYPE']

    TEACHER_USE_FIRST_CODE_FOR_REC = 'TEACHER_USE_FIRST_CODE_FOR_REC' in os.environ
    TEACHER_EXTRA_SA_BEFORE_MAP = 'TEACHER_EXTRA_SA_BEFORE_MAP' in os.environ
    
else:
    USE_STAGE1_AS_TEACHER = False