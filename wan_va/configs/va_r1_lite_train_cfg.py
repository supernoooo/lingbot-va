# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_r1_lite_cfg import va_r1_lite_cfg


va_r1_lite_train_cfg = EasyDict(__name__='Config: VA R1 Lite train')
va_r1_lite_train_cfg.update(va_r1_lite_cfg)

# Point this to the R1 Lite joint LeRobot dataset root after conversion and
# latent extraction. Expected structure follows LingBot-VA's latent LeRobot
# loader:
#   dataset_path/
#     meta/episodes.jsonl        # with action_config
#     data/...
#     videos/... or image payloads referenced by data parquet
#     latents/...
#     empty_emb.pt
va_r1_lite_train_cfg.dataset_path = '/path/to/your/dataset'
va_r1_lite_train_cfg.empty_emb_path = os.path.join(
    va_r1_lite_train_cfg.dataset_path, 'empty_emb.pt'
)

va_r1_lite_train_cfg.enable_wandb = True
va_r1_lite_train_cfg.load_worker = 16
va_r1_lite_train_cfg.save_interval = 1000
va_r1_lite_train_cfg.gc_interval = 50
va_r1_lite_train_cfg.cfg_prob = 0.1

# Training parameters. Keep conservative defaults; tune after a small
# overfitting/smoke run on R1 Lite data.
va_r1_lite_train_cfg.learning_rate = 1e-5
va_r1_lite_train_cfg.beta1 = 0.9
va_r1_lite_train_cfg.beta2 = 0.95
va_r1_lite_train_cfg.weight_decay = 0.1
va_r1_lite_train_cfg.warmup_steps = 10
va_r1_lite_train_cfg.batch_size = 1
va_r1_lite_train_cfg.gradient_accumulation_steps = 1
va_r1_lite_train_cfg.num_steps = 50000
