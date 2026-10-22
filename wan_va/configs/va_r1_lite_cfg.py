# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg


va_r1_lite_cfg = EasyDict(__name__='Config: VA R1 Lite')
va_r1_lite_cfg.update(va_shared_cfg)

# Base checkpoint to initialize from. For R1 Lite post-training, prefer starting
# from lingbot-va-base or a compatible robotwin checkpoint, then replace norm_stat
# with statistics computed from R1 Lite data.
va_r1_lite_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"

# R1 Lite follows the same three-camera layout used by LingBot-VA Robotwin:
#   Galaxea head camera       -> observation.images.cam_high
#   Galaxea left D405 wrist   -> observation.images.cam_left_wrist
#   Galaxea right D405 wrist  -> observation.images.cam_right_wrist
#
# The adapter/dataset converter is responsible for renaming Galaxea keys such as
# observation.images.head_rgb / left_wrist_rgb / right_wrist_rgb to these keys.
va_r1_lite_cfg.obs_cam_keys = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]

# Use the Robotwin T-shape latent layout because R1 Lite also has one high/head
# view plus two wrist views. In _encode_obs this means:
#   cam_high at height x width
#   wrist cameras at half height x half width, tiled together with cam_high
va_r1_lite_cfg.env_type = 'robotwin_tshape'
va_r1_lite_cfg.height = 256
va_r1_lite_cfg.width = 320
va_r1_lite_cfg.patch_size = (1, 2, 2)
va_r1_lite_cfg.attn_window = 72
va_r1_lite_cfg.frame_chunk_size = 2

# LingBot-VA keeps a 30-channel action container. R1 Lite's verified
# GalaxeaManipSim path is joint-position control, so this config uses 14
# effective channels:
#   14:20  left arm joints 6
#   28     left gripper
#   21:27  right arm joints 6
#   29     right gripper
#
# Channels 20 and 27 are left unused because the shared 30-channel container
# reserves 7 joints per arm, while R1 Lite has 6 arm joints per side.
va_r1_lite_cfg.action_dim = 30
va_r1_lite_cfg.action_representation = 'joint'
va_r1_lite_cfg.action_per_frame = 16
va_r1_lite_cfg.used_action_channel_ids = (
    list(range(14, 20)) + [28] + list(range(21, 27)) + [29]
)
inverse_used_action_channel_ids = [
    len(va_r1_lite_cfg.used_action_channel_ids)
] * va_r1_lite_cfg.action_dim
for i, j in enumerate(va_r1_lite_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_r1_lite_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Inference defaults. These mirror Robotwin until R1 Lite-specific tuning is
# available.
va_r1_lite_cfg.guidance_scale = 5
va_r1_lite_cfg.action_guidance_scale = 1
va_r1_lite_cfg.num_inference_steps = 25
va_r1_lite_cfg.video_exec_step = -1
va_r1_lite_cfg.action_num_inference_steps = 50
va_r1_lite_cfg.snr_shift = 5.0
va_r1_lite_cfg.action_snr_shift = 1.0

va_r1_lite_cfg.action_norm_method = 'quantiles'

# These ranges are computed from the provided
# GalaxeaManipSim/datasets/R1LiteBlocksStackEasy/final_galaxea/meta/
# episodes_stats.jsonl metadata by taking global per-channel min/max across 100
# episodes. They are a conservative replacement for the old placeholder values.
# Once the full parquet/h5 action data is available, recompute true q01/q99
# quantiles from raw action rows and replace these min/max-derived values.
va_r1_lite_cfg.norm_stat = {
    "q01": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.042811513, 1.566229463, -2.561428308, -0.0001625751029,
        -0.006231058389, -0.03265309334, 0.0, -4.99510088e-06, 1.569610834, -2.546894312,
        -1.682748696e-08, -0.00630826503, -0.6818361282, 0.0, 0.0, 0.0,
    ],
    "q99": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 2.119221335e-06, 2.69636941, -0.9596404433, 1.483970046,
        0.000737284252, 0.5993646979, 0.0, 1.046555519, 2.695877075, -0.8879969716,
        1.47528553, 0.0002959713165, 1.765985303e-06, 0.0, 0.05000000075, 0.05000000075,
    ],
}

# Metadata for adapters/converters. LingBot-VA core does not consume these
# fields directly, but keeping them here makes the intended R1 Lite mapping
# explicit.
va_r1_lite_cfg.robot_type = 'r1_lite'
va_r1_lite_cfg.galaxea_image_key_map = {
    'observation.images.head_rgb': 'observation.images.cam_high',
    'observation.images.left_wrist_rgb': 'observation.images.cam_left_wrist',
    'observation.images.right_wrist_rgb': 'observation.images.cam_right_wrist',
}
va_r1_lite_cfg.galaxea_action_keys = [
    'action.left_arm_joints',
    'action.left_gripper',
    'action.right_arm_joints',
    'action.right_gripper',
]
