"""OpenPi RL Token Policy: frozen PI0Pytorch backbone + trainable RLTokenAutoencoder + TD3 heads."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch
from torch import Tensor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.rl_token.rl_token import (
    RLTokenAutoencoder,
    RLTokenConfig,
    reconstruction_loss,
)


@dataclasses.dataclass
class OpenPiRLTokenConfig:
    """Config for OpenPiRLTokenPolicy.

    pi0_config: passed to PI0Pytorch (can be None for smoke tests with mocked backbone)
    """

    pi0_config: Any = None
    hidden_dim: int = 2048          # must match PI0Pytorch prefix_output last dim
    rl_token_dim: int = 256
    rl_token_encoder_layers: int = 2
    rl_token_decoder_layers: int = 2
    rl_token_num_heads: int = 8
    rl_token_max_seq_len: int = 512
    rl_token_dropout: float = 0.1
    actor_hidden_dims: tuple = (512, 256)
    critic_hidden_dims: tuple = (512, 256)
    action_horizon: int = 5
    action_dim: int = 7
    recon_loss_coef: float = 0.1


class OpenPiRLTokenPolicy(torch.nn.Module, BasePolicy):
    """Frozen openpi backbone + trainable RLTokenAutoencoder + actor/critic MLP heads.

    Implements BasePolicy.td3_forward() and td3_q_forward() for TD3 training.
    The PI0Pytorch backbone is frozen; only rl_token_autoencoder, actor_head,
    and critic_head_1/2 are trained.
    """

    def __init__(self, config: OpenPiRLTokenConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        rl_cfg = RLTokenConfig(
            hidden_dim=config.hidden_dim,
            rl_token_dim=config.rl_token_dim,
            max_seq_len=config.rl_token_max_seq_len,
            encoder_layers=config.rl_token_encoder_layers,
            decoder_layers=config.rl_token_decoder_layers,
            num_heads=config.rl_token_num_heads,
            dropout=config.rl_token_dropout,
        )

        self.rl_token_autoencoder = RLTokenAutoencoder(rl_cfg)
        self.actor_head = ValueHead(
            input_dim=config.rl_token_dim,
            hidden_sizes=config.actor_hidden_dims,
            output_dim=config.action_horizon * config.action_dim,
            activation="relu",
            bias_last=True,
        )
        self.critic_head_1 = ValueHead(
            input_dim=config.rl_token_dim + config.action_dim,
            hidden_sizes=config.critic_hidden_dims,
            output_dim=1,
            activation="relu",
        )
        self.critic_head_2 = ValueHead(
            input_dim=config.rl_token_dim + config.action_dim,
            hidden_sizes=config.critic_hidden_dims,
            output_dim=1,
            activation="relu",
        )

        # Target networks — worker calls soft_update_target_model to keep them in sync
        self.target_rl_token_autoencoder = copy.deepcopy(self.rl_token_autoencoder)
        self.target_actor_head = copy.deepcopy(self.actor_head)
        self.target_critic_head_1 = copy.deepcopy(self.critic_head_1)
        self.target_critic_head_2 = copy.deepcopy(self.critic_head_2)
        for p in (
            list(self.target_rl_token_autoencoder.parameters())
            + list(self.target_actor_head.parameters())
            + list(self.target_critic_head_1.parameters())
            + list(self.target_critic_head_2.parameters())
        ):
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # BasePolicy abstract methods
    # ------------------------------------------------------------------

    def default_forward(self, **kwargs):
        return self.predict_action_batch(**kwargs)

    def predict_action_batch(self, obs, **kwargs):
        """Rollout inference: encode prefix → actor head → actions."""
        prefix_output, _, _ = self._build_prefix_cache_from_obs(obs)
        rl_token = self.rl_token_autoencoder.encoder(prefix_output)
        return self._decode_action(rl_token, use_target=False)

    # ------------------------------------------------------------------
    # TD3 interface (called by TD3Algorithm via BasePolicy.forward)
    # ------------------------------------------------------------------

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def td3_forward(self, mode: str = "actor", **kwargs):
        if mode == "actor":
            return self._td3_actor_forward(**kwargs)
        elif mode == "critic":
            return self._td3_critic_forward(**kwargs)
        raise ValueError(f"Unknown mode: {mode}")

    def td3_q_forward(self, rl_state: Tensor, action: Tensor, **kwargs):
        """Direct Q-value computation used during actor update."""
        return self._compute_q(rl_state, action, use_target=False)

    # ------------------------------------------------------------------
    # Target network methods (called by TD3Algorithm on unwrapped policy)
    # ------------------------------------------------------------------

    def target_actor_forward(self, visual_feat, robot_state, ref_action, **kwargs):
        prefix_output = self._extract_prefix_from_visual_feat(visual_feat)
        rl_token = self.target_rl_token_autoencoder.encoder(prefix_output)
        actions = self._decode_action(rl_token, use_target=True)
        return actions, {"rl_state": rl_token}

    def target_critic_forward(self, rl_state: Tensor, action: Tensor, **kwargs):
        return self._compute_q(rl_state, action, use_target=True)

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def compute_recon_loss(self, prefix_output: Tensor, rl_token: Tensor) -> Tensor:
        _, recon = self.rl_token_autoencoder.decoder(rl_token, prefix_output), None
        # Use autoencoder forward to get recon (encoder already ran, reuse token)
        recon = self.rl_token_autoencoder.decoder(rl_token, prefix_output)
        return reconstruction_loss(prefix_output, recon)

    # ------------------------------------------------------------------
    # Backbone freezing
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze all PI0Pytorch parameters; only RL heads remain trainable."""
        trainable_modules = {
            self.rl_token_autoencoder,
            self.actor_head,
            self.critic_head_1,
            self.critic_head_2,
        }
        for name, param in self.named_parameters():
            # Check if param belongs to any trainable module
            is_trainable = any(
                any(p is param for p in m.parameters())
                for m in trainable_modules
            )
            if not is_trainable:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _td3_actor_forward(
        self,
        visual_feat,
        robot_state: Tensor,
        ref_action: Tensor,
        ref_action_dropout_p: float = 0.0,
        use_target: bool = False,
        **kwargs,
    ):
        prefix_output = self._extract_prefix_from_visual_feat(visual_feat)
        encoder = (
            self.target_rl_token_autoencoder.encoder if use_target
            else self.rl_token_autoencoder.encoder
        )
        rl_token = encoder(prefix_output)
        actions = self._decode_action(rl_token, use_target=use_target)
        aux = {"rl_state": rl_token, "prefix_output": prefix_output}
        return actions, aux

    def _td3_critic_forward(
        self,
        rl_state: Tensor,
        action: Tensor,
        use_target: bool = False,
        **kwargs,
    ):
        return self._compute_q(rl_state, action, use_target=use_target)

    def _decode_action(self, rl_token: Tensor, use_target: bool) -> Tensor:
        head = self.target_actor_head if use_target else self.actor_head
        flat = head(rl_token)
        return flat.reshape(flat.shape[0], self.config.action_horizon, self.config.action_dim)

    def _compute_q(self, rl_state: Tensor, action: Tensor, use_target: bool):
        # Flatten action to (B, action_dim) if chunked
        if action.dim() == 3:
            action = action.reshape(action.shape[0], -1)[..., : self.config.action_dim]
        critic_input = torch.cat([rl_state, action], dim=-1)
        if use_target:
            q1 = self.target_critic_head_1(critic_input)
            q2 = self.target_critic_head_2(critic_input)
        else:
            q1 = self.critic_head_1(critic_input)
            q2 = self.critic_head_2(critic_input)
        return q1, q2

    def _extract_prefix_from_visual_feat(self, visual_feat) -> Tensor:
        """Extract prefix tokens from visual_feat.

        In production, visual_feat is the raw observation dict and
        _build_prefix_cache is called here. For smoke tests, visual_feat
        can be a pre-computed tensor of shape (B, L, hidden_dim).
        """
        if isinstance(visual_feat, Tensor):
            return visual_feat
        # Production path: call PI0Pytorch._build_prefix_cache
        return self._build_prefix_cache_from_obs(visual_feat)[0]

    def _build_prefix_cache_from_obs(self, obs):
        """Delegate to PI0Pytorch._build_prefix_cache. Subclasses that inherit
        PI0Pytorch will have this method available."""
        raise NotImplementedError(
            "_build_prefix_cache_from_obs must be implemented by the concrete "
            "subclass that also inherits PI0Pytorch, or mocked in tests."
        )
