import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


IMAGE_KEY_MAP = {
    "observation.images.cam_high": "observation.images.head_rgb",
    "observation.images.cam_left_wrist": "observation.images.left_wrist_rgb",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_rgb",
}

ACTION_KEYS = [
    "action.left_arm_joints",
    "action.left_gripper",
    "action.right_arm_joints",
    "action.right_gripper",
]

USED_ACTION_CHANNEL_IDS = list(range(14, 20)) + [28] + list(range(21, 27)) + [29]
MIN_ACTION_NORM_RANGE = 0.05


def read_jsonl(path: Path):
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def data_file_for_episode(root: Path, episode_index: int) -> Path:
    return root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def load_episode(root: Path, episode_index: int):
    return load_dataset(
        "parquet",
        data_files=str(data_file_for_episode(root, episode_index)),
        split="train",
    )


def build_action_config(length: int, action_text: str, window_frames: int):
    # Keep full-episode mode as the default/original route. Use
    # --window-frames 0 when a larger GPU can encode and train whole episodes.
    if window_frames <= 0:
        return [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": action_text,
                "skill": "pnp",
            }
        ]

    # Current R1 Lite sanity route: split each episode into short windows so
    # Wan VAE latent extraction fits on 40GB GPUs. This is a data-prep choice,
    # not a replacement for the full-episode training path above.
    return [
        {
            "start_frame": start_frame,
            "end_frame": start_frame + window_frames,
            "action_text": action_text,
            "skill": "pnp",
        }
        for start_frame in range(0, length - window_frames + 1, window_frames)
    ]


def add_action_config(root: Path, action_text: str, window_frames: int):
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    changed = False
    for ep in episodes:
        action_config = build_action_config(ep["length"], action_text, window_frames)
        if ep.get("action_config") != action_config:
            ep["action_config"] = action_config
            changed = True
    if changed:
        write_jsonl(episodes_path, episodes)
    return episodes


def row_action_14(row):
    parts = []
    for key in ACTION_KEYS:
        value = row[key]
        if isinstance(value, list):
            parts.extend(value)
        else:
            parts.append(value)
    return np.asarray(parts, dtype=np.float32)


def action_14_to_30(action_14: np.ndarray):
    action_30 = np.zeros((action_14.shape[0], 30), dtype=np.float32)
    action_30[:, USED_ACTION_CHANNEL_IDS] = action_14
    return action_30


def compute_action_norm(root: Path, episodes):
    all_actions = []
    for ep in tqdm(episodes, desc="Computing action stats"):
        ds = load_episode(root, ep["episode_index"])
        action_14 = np.stack([row_action_14(row) for row in ds], axis=0)
        all_actions.append(action_14_to_30(action_14))
    actions = np.concatenate(all_actions, axis=0)
    # For R1 Lite PnP sanity training, min/max is less aggressive than q01/q99:
    # several joint channels are nearly constant, and quantile ranges can become
    # tiny enough to amplify actions into very large normalized targets.
    q01 = actions.min(axis=0).astype(np.float32)
    q99 = actions.max(axis=0).astype(np.float32)
    ranges = q99 - q01
    for channel_id in USED_ACTION_CHANNEL_IDS:
        if ranges[channel_id] < MIN_ACTION_NORM_RANGE:
            center = (q01[channel_id] + q99[channel_id]) / 2.0
            q01[channel_id] = center - MIN_ACTION_NORM_RANGE / 2.0
            q99[channel_id] = center + MIN_ACTION_NORM_RANGE / 2.0
    out_path = root / "meta" / "r1_lite_action_norm.json"
    out_path.write_text(json.dumps({"q01": q01.astype(float).tolist(), "q99": q99.astype(float).tolist()}, indent=2))
    return out_path


def encode_prompt(tokenizer, text_encoder, text: str, device, dtype, max_sequence_length=512):
    text_inputs = tokenizer(
        [prompt_clean(text)],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device="cpu")
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
            )
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds[0]


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    return ((latents.float() - latents_mean) * latents_std).to(latents)


def _video_tensor_from_frames(frames, height, width):
    video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
    video = F.interpolate(
        video,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).unsqueeze(0)
    return video / 255.0 * 2.0 - 1.0


def _latent_payload_from_mu(mu_norm, video, height, width):
    latent = rearrange(mu_norm[0].cpu(), "c f h w -> (f h w) c").to(torch.bfloat16)
    return {
        "latent": latent,
        "latent_num_frames": int(mu_norm.shape[2]),
        "latent_height": int(mu_norm.shape[3]),
        "latent_width": int(mu_norm.shape[4]),
        "video_num_frames": int(video.shape[2]),
        "video_height": int(height),
        "video_width": int(width),
    }


def encode_video_official(vae, frames, height, width, device, dtype):
    video = _video_tensor_from_frames(frames, height, width).to(device=device, dtype=dtype)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device)
    latents_std = torch.tensor(vae.config.latents_std, device=device)
    with torch.no_grad():
        enc_out = vae.encode(video)
        mu = enc_out.latent_dist.mean
        mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)
    payload = _latent_payload_from_mu(mu_norm, video, height, width)
    del enc_out, mu, mu_norm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def encode_video_streaming(vae_wrapper, vae, frames, height, width, device, dtype, chunk_frames):
    vae_wrapper.clear_cache()
    video = _video_tensor_from_frames(frames, height, width)
    latent_chunks = []
    latents_mean = torch.tensor(vae.config.latents_mean, device=device)
    latents_std = torch.tensor(vae.config.latents_std, device=device)
    with torch.no_grad():
        for start in range(0, video.shape[2], chunk_frames):
            video_chunk = video[:, :, start : start + chunk_frames].to(
                device=device, dtype=dtype
            )
            enc_out = vae_wrapper.encode_chunk(video_chunk)
            mu, _ = torch.chunk(enc_out, 2, dim=1)
            mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)
            latent_chunks.append(mu_norm.cpu())
            del video_chunk, enc_out, mu, mu_norm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    mu_norm = torch.cat(latent_chunks, dim=2)
    payload = _latent_payload_from_mu(mu_norm, video, height, width)
    del mu_norm
    return payload


def extract_latents(
    root: Path,
    episodes,
    checkpoint: Path,
    action_text: str,
    device: str,
    dtype,
    chunk_frames: int,
    encode_mode: str,
    max_episodes: int | None,
):
    tokenizer = load_tokenizer(checkpoint / "tokenizer")
    text_encoder = load_text_encoder(checkpoint / "text_encoder", dtype, device)
    vae = load_vae(checkpoint / "vae", dtype, device)
    vae_wrapper = WanVAEStreamingWrapper(vae) if encode_mode == "streaming" else None

    text_emb = encode_prompt(tokenizer, text_encoder, action_text, device, dtype)
    empty_emb = encode_prompt(tokenizer, text_encoder, "", device, dtype)
    torch.save(empty_emb, root / "empty_emb.pt")
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    episodes_to_extract = episodes[:max_episodes] if max_episodes is not None else episodes
    for ep in tqdm(episodes_to_extract, desc="Extracting latents"):
        episode_index = ep["episode_index"]
        ds = load_episode(root, episode_index)
        for acfg in ep["action_config"]:
            start_frame = acfg["start_frame"]
            end_frame = acfg["end_frame"]
            rows = [ds[i] for i in range(start_frame, end_frame)]
            frame_ids = list(range(start_frame, end_frame))
            for out_key, src_key in IMAGE_KEY_MAP.items():
                out_dir = root / "latents" / "chunk-000" / out_key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                if out_file.exists():
                    continue
                height, width = (256, 320) if out_key.endswith("cam_high") else (128, 160)
                frames = [np.asarray(row[src_key].convert("RGB")) for row in rows]
                if encode_mode == "official":
                    payload = encode_video_official(vae, frames, height, width, device, dtype)
                else:
                    payload = encode_video_streaming(
                        vae_wrapper, vae, frames, height, width, device, dtype, chunk_frames
                    )
                payload.update(
                    {
                        "text_emb": text_emb,
                        "text": acfg.get("action_text", action_text),
                        "frame_ids": frame_ids,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "fps": 15,
                        "ori_fps": 15,
                    }
                )
                torch.save(payload, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--action-text",
        default="pick the object and place it on the blue target",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--vae-chunk-frames", type=int, default=2)
    parser.add_argument("--vae-encode-mode", default="official", choices=["official", "streaming"])
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-action-stats", action="store_true")
    parser.add_argument("--skip-latents", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    episodes = add_action_config(root, args.action_text, args.window_frames)
    if not args.skip_action_stats:
        norm_path = compute_action_norm(root, episodes)
        print(f"Wrote action norm stats: {norm_path}")

    if not args.skip_latents:
        extract_latents(
            root,
            episodes,
            Path(args.checkpoint),
            args.action_text,
            args.device,
            dtype,
            args.vae_chunk_frames,
            args.vae_encode_mode,
            args.max_episodes,
        )
        print(f"Wrote latents under: {root / 'latents'}")
        print(f"Wrote empty embedding: {root / 'empty_emb.pt'}")


if __name__ == "__main__":
    main()
