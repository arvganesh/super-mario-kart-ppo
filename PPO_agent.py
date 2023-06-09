# Code adapted from https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py

import argparse
import datetime
import os
import pprint

import numpy as np
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger, LazyLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from configure_env import make_retro_env
from network import PolicyNet, layer_init


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MarioKart-Snes")
    parser.add_argument("--custom-integration-path", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="scenario")
    parser.add_argument("--info", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=646269)
    parser.add_argument("--scale-obs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gae-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--recompute-adv", type=bool, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb", "lazy"],
    )
    parser.add_argument("--wandb-project", type=str, default="kart-snes.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)

    # TODO: Integrate Intrinsic Curiosity Module
    # parser.add_argument(
    #     "--icm-lr-scale",
    #     type=float,
    #     default=0.,
    #     help="use intrinsic curiosity module with this lr scale"
    # )
    # parser.add_argument(
    #     "--icm-reward-scale",
    #     type=float,
    #     default=0.01,
    #     help="scaling factor for intrinsic curiosity reward"
    # )
    # parser.add_argument(
    #     "--icm-forward-loss-weight",
    #     type=float,
    #     default=0.2,
    #     help="weight for the forward model loss in ICM"
    # )

    return parser.parse_args()


def test_ppo(args=get_args()):
    train_envs, test_envs = make_retro_env(
        args.task,
        args.training_num,
        args.test_num,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        scenario=args.scenario,
        info=args.info,
        custom_integration_path=args.custom_integration_path,
        max_episode_steps=args.max_episode_steps,
        video_path=args.video_path,
    )

    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Device:", args.device)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Define model. TODO: Move this outside of Tianshou wrappers.
    actor_net = PolicyNet(device=args.device).to(device=args.device)
    critic_net = PolicyNet(device=args.device).to(device=args.device)
    actor = Actor(actor_net, args.action_shape, [args.hidden_size], device=args.device, softmax_output=False) # softmax_output is False b/c torch.distributions.Categorical takes in log-probs.
    critic = Critic(critic_net, [args.hidden_size], device=args.device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(
        actor_critic.parameters(), lr=args.lr, eps=1e-5
    )

    # Last Layer Initialization
    layer_init(actor.last.model[-1], std=0.01)
    layer_init(critic.last.model[-1], std=1.0)

    print("Num Parameters: ")
    print("Actor: ", sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print("Critic: ", sum(p.numel() for p in critic.parameters() if p.requires_grad))
    print("Total: ", sum(p.numel() for p in actor_critic.parameters() if p.requires_grad))

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # Policy = Categorical Distribution parameterized by logits
    def dist(p):
        return torch.distributions.Categorical(logits=p) # logits are interpreted as unnormalized log-probs.

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        max_batchsize=args.gae_batch_size,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, 
        reward_normalization=args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=train_envs.action_space[0],
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)

    # TODO: Integrate Intrinsic Curiosity
    # if args.icm_lr_scale > 0:
    #     feature_net = DQN(
    #         *args.state_shape, args.action_shape, args.device, features_only=True
    #     )
    #     action_dim = np.prod(args.action_shape)
    #     feature_dim = feature_net.output_dim
    #     icm_net = IntrinsicCuriosityModule(
    #         feature_net.net,
    #         feature_dim,
    #         action_dim,
    #         hidden_sizes=[args.hidden_size],
    #         device=args.device,
    #     )
    #     icm_optim = torch.optim.Adam(icm_net.parameters(), lr=args.lr)
    #     policy = ICMPolicy(
    #         policy, icm_net, icm_optim, args.icm_lr_scale, args.icm_reward_scale,
    #         args.icm_forward_loss_weight
    #     ).to(args.device)

    # load a previous policy
    if args.resume_path:
        # load from existing checkpoint
        print(f"Loading agent under {args.resume_path}")
        if os.path.exists(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frame_stack,
    )
    
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # Logger
    if not args.watch:
        if args.logger == "wandb":

            logger = WandbLogger(
                save_interval=5,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project
            )

            wandb.watch(actor, log="all", log_freq=10)
            wandb.watch(critic, log="all", log_freq=10)
            

        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))

        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        elif args.logger == "wandb": # wandb
            logger.load(writer)
        else:
            logger = LazyLogger()

    def save_best_fn(policy):
        print("Saving best performer at:", str(os.path.join(log_path, "policy.pth")))
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        print("Saving checkpoint at:", str(ckpt_path))
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    # test train_collector and start filling replay buffer
    print("Start training ...")
    train_collector.collect(n_step=args.batch_size)
    print("Train collector test done.")

    # Trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch, # number of epochs
        args.step_per_epoch, # number of steps per epoch
        args.repeat_per_collect, # how many times we update the policy w/ the same batch
        args.test_num, # episodes per test
        args.batch_size, # size of batch we update the policy on - "mini_batch" size
        step_per_collect=args.step_per_collect, # number of steps per collect call, equivalent to "batch_size" in other vectorized PPO implementations
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn
    )

    pprint.pprint(result)


if __name__ == "__main__":
    test_ppo(get_args())