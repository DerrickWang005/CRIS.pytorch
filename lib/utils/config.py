from detectron2.config import CfgNode as CN


def add_cris_config(cfg):
    """
    Add config for uniovseg.
    """
    cfg.INPUT.DATASET_NAME = "refer"
    cfg.INPUT.TRAIN_ROOT = "/datasets/refcoco"
    cfg.INPUT.TRAIN_NAME = "refcoco"
    cfg.INPUT.REFER_SPLIT = "unc"
    cfg.INPUT.DATA_SPLIT = "train"
    cfg.INPUT.DATA_TEST_SPLIT = "val"
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.IGNORE_LABEL = 0
    cfg.INPUT.POS_REPEAT = 5
    cfg.INPUT.CROP_SIZE = 640

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.SOLVER.POLY_LR_POWER = 0.9

    cfg.MODEL.META_ARCHITECTURE = "CRIS"
    cfg.MODEL.BACKBONE.NAME = "CLIP"
    cfg.MODEL.BACKBONE.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.BACKBONE.CLIP_PRETRAINED_WEIGHTS = "/workspace/pretrains/convnext_large_d_320.laion2B-s29B-b131K-ft-soup.pth"

    cfg.MODEL.CRIS = CN()
    cfg.MODEL.CRIS.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.CRIS.TRANSFORMER_DECODER_NAME = "MultiScaleMaskDecoder"
    cfg.MODEL.CRIS.IN_FEATURES = [
        "res2",
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.CRIS.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.CRIS.MASK_DIM = 256
    cfg.MODEL.CRIS.CLS_DIM = 1024
    cfg.MODEL.CRIS.CONVS_DIM = 256
    cfg.MODEL.CRIS.NORM = "GN"
    cfg.MODEL.CRIS.NHEADS = 8
    cfg.MODEL.CRIS.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.CRIS.COMMON_STRIDE = 4
    cfg.MODEL.CRIS.EMBED_DIM = 256
    cfg.MODEL.CRIS.ACTIVATION = "relu"
    cfg.MODEL.CRIS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.CRIS.DEC_LAYERS = 7
    cfg.MODEL.CRIS.PRE_NORM = False
    cfg.MODEL.CRIS.ENFORCE_INPUT_PROJ = True
    cfg.MODEL.CRIS.NUM_QUERY = 50
    cfg.MODEL.CRIS.MASK_LAYERS = 3
    cfg.MODEL.CRIS.DROPOUT = 0.0
    cfg.MODEL.CRIS.NUM_CLASSES = 1
    cfg.MODEL.CRIS.DICE_WEIGHT = 5.0
    cfg.MODEL.CRIS.MASK_WEIGHT = 5.0
    cfg.MODEL.CRIS.CLASS_WEIGHT = 2.0
    cfg.MODEL.CRIS.TRAIN_NUM_POINTS = 12544  # 800 * 800 // (8 * 8)
    cfg.MODEL.CRIS.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.CRIS.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.CRIS.DEEP_SUPERVISION = True
    cfg.MODEL.CRIS.CRITERION = "SetCriterion"

    cfg.MODEL.CRIS.TEST = CN()
    cfg.MODEL.CRIS.TEST.VISUALIZE_SAVE = False
