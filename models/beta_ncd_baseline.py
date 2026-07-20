"""BETA-CD with an NCD backbone for student-disjoint evaluation.

This is an independent, protocol-specific adaptation of BETA-CD and its NCD
components in PyAT. The upstream implementation is MIT licensed; provenance
and the exact adaptation boundary are documented in
``docs/third_party/beta_ncd.md``.

Unlike ID-parameterized cognitive-diagnosis models, this module has no student
embedding table. A student's diagonal-Gaussian posterior is initialized from
a shared prior and adapted only from that student's support responses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BETA_NCD_NAME = "BETA-CD (NCD backbone, paper-aligned adaptation)"
BETA_NCD_UPSTREAM_COMMIT = "8e19a3759548bebd39f37a34977b1a2148f3c19c"
BETA_NCD_UPSTREAM_URL = "https://github.com/AyiStar/pyat"


@dataclass(frozen=True)
class BetaNCDConfig:
    """Frozen paper-aligned configuration used by the standalone trainer."""

    num_items: int
    num_concepts: int
    inner_steps: int = 3
    inner_mc_samples: int = 4
    query_mc_samples: int = 4
    kl_weight: float = 1e-4
    inner_lr: float = 0.1
    hidden_dim_1: int = 256
    hidden_dim_2: int = 32
    dropout: float = 0.2
    seed: int = 42

    def architecture_payload(self) -> dict[str, object]:
        """Return the dataset-independent topology used for fingerprinting."""

        payload = asdict(self)
        payload.pop("num_items")
        payload.pop("num_concepts")
        return {
            "model": "beta_ncd_paper_aligned_v2",
            "student_parameterization": "support_adapted_diagonal_gaussian",
            "outer_objective": "query_log_mean_predictive_likelihood",
            "inner_response_reduction": "mc_mean_support_sum",
            "inner_kl_application": "once_per_full_support_step",
            "outer_response_reduction": "joint_query_logmean_no_length_normalization",
            "q_semantics": "union",
            "ncd_hidden_activation": "relu",
            "ncd_item_scalar": "sigmoid_without_x10",
            "positive_linear": "absolute_weight",
            "ncd_weight_init": "xavier_normal",
            "prior_mean_init": "normal_0_1",
            "ncd_bias_init": "torch_linear_default",
            "prior_log_std_init": "uniform_minus4_minus3",
            "upstream_commit": BETA_NCD_UPSTREAM_COMMIT,
            **payload,
        }

    def architecture_fingerprint(self) -> str:
        canonical = json.dumps(
            self.architecture_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdaptedPosterior:
    mean: torch.Tensor
    log_std: torch.Tensor


class PositiveLinear(nn.Linear):
    """NCD monotonic layer using the author's absolute-weight projection."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positive_weight = 2.0 * F.relu(-self.weight) + self.weight
        return F.linear(inputs, positive_weight, self.bias)


class BetaNCDBaseline(nn.Module):
    """BETA-CD Bayesian task adaptation with an NCD response function.

    Global parameters consist of a shared Gaussian prior, item parameters and
    a monotonic NCD interaction network. There is deliberately no student-ID
    parameter. ``adapt`` is the only path from support responses to a local
    student posterior.
    """

    def __init__(self, *, config: BetaNCDConfig, q_matrix: torch.Tensor):
        super().__init__()
        if config.num_items < 1 or config.num_concepts < 1:
            raise ValueError("num_items and num_concepts must be positive.")
        if q_matrix.shape != (config.num_items, config.num_concepts):
            raise ValueError(
                "q_matrix shape must match (num_items, num_concepts): "
                f"got {tuple(q_matrix.shape)}."
            )
        if config.inner_steps != 3:
            raise ValueError("The paper-aligned baseline fixes inner_steps=3.")
        if config.inner_mc_samples != 4 or config.query_mc_samples != 4:
            raise ValueError("The paper-aligned baseline fixes Nt=Nv=4.")

        self.config = config
        self.register_buffer("q_matrix", (q_matrix > 0).to(torch.float32))

        # Shared meta-prior. A held-out student receives a local copy initialized
        # from these two vectors; no posterior is stored in the checkpoint.
        self.prior_mean = nn.Parameter(torch.empty(config.num_concepts))
        self.prior_log_std = nn.Parameter(torch.empty(config.num_concepts))
        self.inner_lrs = nn.Parameter(
            torch.full((config.inner_steps,), float(config.inner_lr))
        )

        self.knowledge_difficulty = nn.Embedding(
            config.num_items, config.num_concepts
        )
        self.item_difficulty = nn.Embedding(config.num_items, 1)
        self.interaction = nn.Sequential(
            PositiveLinear(config.num_concepts, config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            PositiveLinear(config.hidden_dim_1, config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            PositiveLinear(config.hidden_dim_2, 1),
        )
        self._reset_parameters()

        # Evaluation noise is fixed, independent of student identity and query
        # order. This makes checkpoint/prediction hashes auditable.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 104_729)
        self.register_buffer(
            "fixed_inner_noise",
            torch.randn(
                config.inner_steps,
                config.inner_mc_samples,
                config.num_concepts,
                generator=generator,
            ),
        )
        self.register_buffer(
            "fixed_query_noise",
            torch.randn(
                config.query_mc_samples,
                config.num_concepts,
                generator=generator,
            ),
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.prior_mean)
        nn.init.uniform_(self.prior_log_std, a=-4.0, b=-3.0)
        nn.init.xavier_normal_(self.knowledge_difficulty.weight)
        nn.init.xavier_normal_(self.item_difficulty.weight)
        for module in self.interaction:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    @property
    def architecture_fingerprint(self) -> str:
        return self.config.architecture_fingerprint()

    def meta_parameters(self) -> list[nn.Parameter]:
        return [self.prior_mean, self.prior_log_std, self.inner_lrs]

    def item_and_response_parameters(self) -> list[nn.Parameter]:
        meta_ids = {id(parameter) for parameter in self.meta_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in meta_ids
        ]

    @staticmethod
    def gaussian_kl(
        posterior_mean: torch.Tensor,
        posterior_log_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_std: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p) for diagonal Gaussians."""

        q_log_std = posterior_log_std
        p_log_std = prior_log_std
        variance_ratio = torch.exp(2.0 * (q_log_std - p_log_std))
        mean_term = (posterior_mean - prior_mean).square() * torch.exp(
            -2.0 * p_log_std
        )
        return 0.5 * torch.sum(
            variance_ratio + mean_term - 1.0 + 2.0 * (p_log_std - q_log_std)
        )

    def ncd_interaction_input(
        self, latent: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        if item_ids.ndim != 1:
            raise ValueError("item_ids must be a one-dimensional tensor.")
        mastery = torch.sigmoid(latent)
        knowledge_difficulty = torch.sigmoid(
            self.knowledge_difficulty(item_ids)
        )
        # Pinned PyAT ncd.py calls this scalar item_difficulty. Its optional
        # multiplication by ten is commented out in the author implementation.
        item_difficulty = torch.sigmoid(self.item_difficulty(item_ids))
        q_vectors = self.q_matrix[item_ids]
        return (
            item_difficulty
            * (mastery.unsqueeze(0) - knowledge_difficulty)
            * q_vectors
        )

    def _predict_from_latent(
        self, latent: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        interaction_input = self.ncd_interaction_input(latent, item_ids)
        logits = self.interaction(interaction_input).squeeze(-1)
        return torch.sigmoid(logits)

    def predictive_samples(
        self,
        *,
        posterior: AdaptedPosterior,
        item_ids: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.config.num_concepts:
            raise ValueError("noise must have shape [samples, num_concepts].")
        posterior_log_std = posterior.log_std
        latent_samples = posterior.mean.unsqueeze(0) + torch.exp(
            posterior_log_std
        ).unsqueeze(0) * noise
        return torch.stack(
            [self._predict_from_latent(latent, item_ids) for latent in latent_samples],
            dim=0,
        )

    def local_variational_loss(
        self,
        *,
        posterior: AdaptedPosterior,
        support_item_ids: torch.Tensor,
        support_labels: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Eq. (5): MC-mean joint support NLL plus one KL term."""

        sample_probs = self.predictive_samples(
            posterior=posterior,
            item_ids=support_item_ids,
            noise=noise,
        )
        expanded_labels = support_labels.to(sample_probs.dtype).unsqueeze(0).expand_as(
            sample_probs
        )
        response_nll = F.binary_cross_entropy(
            sample_probs,
            expanded_labels,
            reduction="none",
        ).sum(dim=1).mean()
        kl = self.gaussian_kl(
            posterior.mean,
            posterior.log_std,
            self.prior_mean,
            self.prior_log_std,
        )
        return response_nll + self.config.kl_weight * kl

    def adapt(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_labels: torch.Tensor,
        create_graph: bool,
        fixed_noise: bool,
    ) -> AdaptedPosterior:
        """Adapt one local posterior using support responses only."""

        if support_item_ids.ndim != 1 or support_labels.ndim != 1:
            raise ValueError("support_item_ids and support_labels must be vectors.")
        if support_item_ids.numel() != support_labels.numel():
            raise ValueError("Support item and label counts differ.")
        if support_item_ids.numel() == 0:
            raise ValueError("At least one support response is required.")

        mean = self.prior_mean
        log_std = self.prior_log_std
        for step_index in range(self.config.inner_steps):
            if fixed_noise:
                noise = self.fixed_inner_noise[step_index]
            else:
                noise = torch.randn(
                    self.config.inner_mc_samples,
                    self.config.num_concepts,
                    device=support_item_ids.device,
                    dtype=self.prior_mean.dtype,
                )
            posterior = AdaptedPosterior(mean=mean, log_std=log_std)
            inner_objective = self.local_variational_loss(
                posterior=posterior,
                support_item_ids=support_item_ids,
                support_labels=support_labels,
                noise=noise,
            )
            mean_grad, log_std_grad = torch.autograd.grad(
                inner_objective,
                (mean, log_std),
                create_graph=create_graph,
            )
            learning_rate = self.inner_lrs[step_index]
            mean = mean - learning_rate * mean_grad
            log_std = log_std - learning_rate * log_std_grad
            if not create_graph:
                mean = mean.detach().requires_grad_(True)
                log_std = log_std.detach().requires_grad_(True)

        if not create_graph:
            mean = mean.detach()
            log_std = log_std.detach()
        return AdaptedPosterior(mean=mean, log_std=log_std)

    def query_meta_loss(
        self,
        *,
        posterior: AdaptedPosterior,
        query_item_ids: torch.Tensor,
        query_labels: torch.Tensor,
        fixed_noise: bool = False,
    ) -> torch.Tensor:
        """Negative log mean predictive likelihood on the disjoint query set."""

        if query_item_ids.numel() == 0:
            raise ValueError("At least one query response is required.")
        noise = (
            self.fixed_query_noise
            if fixed_noise
            else torch.randn(
                self.config.query_mc_samples,
                self.config.num_concepts,
                device=query_item_ids.device,
                dtype=self.prior_mean.dtype,
            )
        )
        sample_probs = self.predictive_samples(
            posterior=posterior,
            item_ids=query_item_ids,
            noise=noise,
        ).clamp(min=1e-7, max=1.0 - 1e-7)
        labels = query_labels.to(sample_probs.dtype).unsqueeze(0)
        sample_log_likelihood = (
            labels * torch.log(sample_probs)
            + (1.0 - labels) * torch.log1p(-sample_probs)
        ).sum(dim=1)
        log_mean_likelihood = torch.logsumexp(
            sample_log_likelihood, dim=0
        ) - math.log(self.config.query_mc_samples)
        return -log_mean_likelihood

    def predict_query(
        self,
        *,
        posterior: AdaptedPosterior,
        query_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        sample_probs = self.predictive_samples(
            posterior=posterior,
            item_ids=query_item_ids,
            noise=self.fixed_query_noise,
        )
        return sample_probs.mean(dim=0)

    def posterior_mastery(
        self, posterior: AdaptedPosterior
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = posterior.log_std
        latent = posterior.mean.unsqueeze(0) + torch.exp(log_std).unsqueeze(
            0
        ) * self.fixed_query_noise
        mastery_samples = torch.sigmoid(latent)
        return mastery_samples.mean(dim=0), mastery_samples.std(dim=0, unbiased=False)
