#!/usr/bin/env bash
set -euo pipefail

CKPT=checkpoints/pretrained/act_aloha_sim_transfer_cube_human

# migrate the old-format Hub checkpoint into a local directory (only once)
if [ ! -f "$CKPT/policy_preprocessor.json" ]; then
  python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
    --output-dir "$CKPT"
fi

lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 --eval.batch_size=5 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --policy.device=cuda \
  --output_dir=outputs/eval/pretrained_tc_human \
  --job_name=eval_ckp
