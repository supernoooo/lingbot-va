import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

LINGBOT_ROOT = Path(__file__).resolve().parents[2]
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


# Set GALAXEA_ROOT to the GalaxeaManipSim checkout if galaxea_sim is not installed.
GALAXEA_ROOT = os.environ.get("GALAXEA_ROOT")
if GALAXEA_ROOT and GALAXEA_ROOT not in sys.path:
    sys.path.insert(0, GALAXEA_ROOT)


IMAGE_KEY_MAP = {
    "observation.images.cam_high": "rgb_head",
    "observation.images.cam_left_wrist": "rgb_left_hand",
    "observation.images.cam_right_wrist": "rgb_right_hand",
}


def make_env():
    import gymnasium as gym
    import galaxea_sim.envs.robotwin  # noqa: F401 - registers gym envs

    return gym.make(
        "R1LitePickPlaceGeneral-v0",
        controller_type="bimanual_joint_position",
        headless=True,
        obs_mode="image",
        control_freq=15,
    )


def format_obs(env_obs):
    images = env_obs["upper_body_observations"]
    return {
        out_key: np.asarray(images[src_key], dtype=np.uint8)
        for out_key, src_key in IMAGE_KEY_MAP.items()
    }


def iter_joint_actions(model_action, skip_first_frame=False, max_substeps=None):
    """Expand server output [14, frame_chunk, action_per_frame] into env.step actions."""
    action = np.asarray(model_action, dtype=np.float32)
    if action.ndim != 3 or action.shape[0] != 14:
        raise ValueError(f"Expected action shape [14, F, H], got {action.shape}")

    count = 0
    start_frame = 1 if skip_first_frame and action.shape[1] > 1 else 0
    for frame_i in range(start_frame, action.shape[1]):
        for substep_i in range(action.shape[2]):
            if max_substeps is not None and count >= max_substeps:
                return
            yield frame_i, substep_i, action[:, frame_i, substep_i]
            count += 1


def run_episode(policy, args, episode_idx):
    env = make_env()
    obs, reset_info = env.reset(seed=args.seed + episode_idx)
    prompt = obs["language_instruction"]
    policy.infer({"reset": True, "prompt": prompt, "save_visualization": args.save_visualization})

    success = False
    total_env_steps = 0
    last_info = {}
    action_history = []

    first_chunk = True
    try:
        for chunk_idx in range(args.max_chunks):
            if args.reset_each_chunk and chunk_idx > 0:
                policy.infer({"reset": True, "prompt": prompt, "save_visualization": args.save_visualization})

            request_obs = format_obs(obs)
            ret = policy.infer(
                {
                    "obs": request_obs,
                    "prompt": prompt,
                    "save_visualization": args.save_visualization,
                    "stateless": args.stateless_chunk,
                }
            )
            action = np.asarray(ret["action"], dtype=np.float32)
            key_frame_obs = []

            action_per_key_frame = max(1, action.shape[2] // 4)
            for _, substep_i, joint_action in iter_joint_actions(
                action,
                skip_first_frame=(args.skip_first_frame and first_chunk),
                max_substeps=args.max_substeps_per_chunk,
            ):
                obs, reward, terminated, truncated, info = env.step(joint_action)
                total_env_steps += 1
                last_info = info
                action_history.append(joint_action)
                success = bool(info.get("success", False) or terminated)

                if (substep_i + 1) % action_per_key_frame == 0:
                    key_frame_obs.append(format_obs(obs))
                if success or truncated or total_env_steps >= args.max_env_steps:
                    break

            if args.compute_kv_cache and key_frame_obs:
                policy.infer(
                    {
                        "obs": key_frame_obs,
                        "state": action,
                        "compute_kv_cache": True,
                        "save_visualization": args.save_visualization,
                    }
                )

            first_chunk = False
            if success or total_env_steps >= args.max_env_steps:
                break
    finally:
        env.close()

    return {
        "episode": episode_idx,
        "seed": args.seed + episode_idx,
        "prompt": prompt,
        "success": bool(success),
        "env_steps": int(total_env_steps),
        "action_steps": len(action_history),
        "reset_info": reset_info,
        "last_info": last_info,
    }


def run(args):
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    results = [run_episode(policy, args, i) for i in range(args.num_episodes)]
    output = {
        "host": args.host,
        "port": args.port,
        "num_episodes": args.num_episodes,
        "success_count": sum(int(r["success"]) for r in results),
        "success_rate": sum(int(r["success"]) for r in results) / max(1, len(results)),
        "max_chunks": args.max_chunks,
        "max_env_steps": args.max_env_steps,
        "compute_kv_cache": args.compute_kv_cache,
        "stateless_chunk": args.stateless_chunk,
        "episodes": results,
    }
    print(json.dumps(output, indent=2, default=lambda x: np.asarray(x).tolist()))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29636)
    parser.add_argument("--output", default="log/r1lite_pnp_rollout_eval.json")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=120)
    parser.add_argument("--max-substeps-per-chunk", type=int, default=None)
    parser.add_argument("--compute-kv-cache", action="store_true")
    parser.add_argument("--stateless-chunk", action="store_true")
    parser.add_argument("--reset-each-chunk", action="store_true")
    parser.add_argument("--skip-first-frame", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
