from network import create_q_network
from env import n_actions , device
from torch import optim as O
from torch.optim.lr_scheduler import LambdaLR
import math
from config import *
from replay_memory import create_replay_memory
import logging
import torch

logger = logging.getLogger(__name__)

policy_net = create_q_network(
    arch=ARCH,
    n_actions=n_actions
)

if PRELOAD_WEIGHT is not None :
    policy_net.load_state_dict(torch.load(PRELOAD_WEIGHT,map_location=torch.device("cpu"))['model'])
    print("model weight loaded from {}".format(PRELOAD_WEIGHT))
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
else : 
    logger.info("model weight loaded from {}".format(PRELOAD_WEIGHT))
    print("No initital weight provided, fall back to random weight")

policy_net = policy_net.to(device)
target_net = create_q_network(
    arch=ARCH,
    n_actions=n_actions
).to(device)
target_net.load_state_dict(policy_net.state_dict())

if METHOD == 'STARFORMER':
    optimizer = O.AdamW(
        policy_net.parameters(),
        lr=STARFORMER_LR,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    def _starformer_lr_lambda(step: int) -> float:
        warmup = max(1, STARFORMER_WARMUP_STEPS)
        if step < warmup:
            return float(step) / float(warmup)
        if MAX_STEPS is None or MAX_STEPS <= warmup:
            return 1.0
        progress = (step - warmup) / float(MAX_STEPS - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=_starformer_lr_lambda)
else:
    optimizer = O.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    scheduler = None

memory = create_replay_memory(SAMPLING_METHOD, MEMORY_CAP, device)