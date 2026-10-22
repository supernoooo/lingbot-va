import json
import os

from easydict import EasyDict

from .va_r1_lite_cfg import va_r1_lite_cfg


va_r1_lite_pnp_train_cfg = EasyDict(__name__="Config: VA R1 Lite PnP train")
va_r1_lite_pnp_train_cfg.update(va_r1_lite_cfg)

va_r1_lite_pnp_train_cfg.dataset_path = "/path/to/your/dataset"
va_r1_lite_pnp_train_cfg.empty_emb_path = os.path.join(
    va_r1_lite_pnp_train_cfg.dataset_path, "empty_emb.pt"
)

norm_path = os.path.join(
    va_r1_lite_pnp_train_cfg.dataset_path,
    "meta",
    "r1_lite_action_norm.json",
)
# This config targets the current 2-frame R1 Lite PnP sanity dataset. The data
# root can still be regenerated in full-episode mode with
# prepare_r1lite_lerobot.py --window-frames 0 for larger-GPU training.
if os.path.exists(norm_path):
    with open(norm_path, "r") as f:
        va_r1_lite_pnp_train_cfg.norm_stat = json.load(f)

va_r1_lite_pnp_train_cfg.enable_wandb = False
va_r1_lite_pnp_train_cfg.num_init_worker = 1
va_r1_lite_pnp_train_cfg.load_worker = 4
va_r1_lite_pnp_train_cfg.save_interval = 100
va_r1_lite_pnp_train_cfg.gc_interval = 10
va_r1_lite_pnp_train_cfg.cfg_prob = 0.1
va_r1_lite_pnp_train_cfg.short_window_action_pad_after = True

va_r1_lite_pnp_train_cfg.learning_rate = 1e-5
va_r1_lite_pnp_train_cfg.beta1 = 0.9
va_r1_lite_pnp_train_cfg.beta2 = 0.95
va_r1_lite_pnp_train_cfg.weight_decay = 0.1
va_r1_lite_pnp_train_cfg.warmup_steps = 10
va_r1_lite_pnp_train_cfg.batch_size = 1
va_r1_lite_pnp_train_cfg.gradient_accumulation_steps = 1
va_r1_lite_pnp_train_cfg.num_steps = 100
