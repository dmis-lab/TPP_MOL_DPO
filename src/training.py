from __future__ import annotations

import argparse
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from transformers import FlaxPreTrainedModel
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModule

from utils import FlaxGPT2LMHeadModelWrapper


class TrainState(train_state.TrainState):
    jacbuf: Array
    jacmom: Array
    lambuf: Array | None
    lammom: Array
    lamreg: Array
    lampref: Array


class TrainModule(nn.Module):
    act: FlaxGPT2LMHeadModule
    ref: FlaxGPT2LMHeadModule
    penalty_beta: float = 0.1

    def _compute_logprobs(self, model: nn.Module, tokens: Array, mask: Array) -> Array:
        logits = model(tokens, mask, jnp.cumsum(mask, axis=-1) - 1).logits
        logprobs = nn.log_softmax(logits[:, :-1, :].astype(jnp.float32))
        return jnp.take_along_axis(logprobs, tokens[:, 1:, None], axis=-1)[..., 0]

    def __call__(self, tokens: Array, mask: Array, labels: Array) -> ArrayTree:
        logp_act = self._compute_logprobs(self.act, tokens, mask)
        logp_ref = self._compute_logprobs(self.ref, tokens, mask)
        logits = (mask[:, 1:] * (logp_act - logp_ref)).sum(-1)

        # Gather the difference of the log probability between policy and reference
        # models and the ground truth reward of each sequence across all devices.
        logits, labels = jax.lax.all_gather((logits, labels), "batch", tiled=True)
        logits = logits[:, None] - logits[None, :]
        sign = jnp.sign(labels[:, None] - labels[None, :])

        loss = -nn.log_sigmoid(sign * self.penalty_beta * logits[:, :, None])
        loss = loss.mean((0, 1))

        accuracy = jnp.sign(logits[:, :, None]) == sign
        accuracy = (accuracy * jnp.abs(sign)).sum() / jnp.abs(sign).sum()
        return {"loss": loss, "accuracy": accuracy}

    def generate(
        self, tokens: Array, sample_rng: PRNGKey, max_length: int
    ) -> tuple[Array, ArrayTree]:
        outputs = FlaxGPT2LMHeadModelWrapper(self.act.config, self.act).generate(
            tokens,
            prng_key=sample_rng,
            params=self.act.variables["params"],
            do_sample=True,
            temperature=1.0,
            max_length=max_length,
        )
        outputs = outputs.sequences
        mask = jnp.cumsum(outputs == self.act.config.eos_token_id, axis=-1) < 1

        logp_act = self._compute_logprobs(self.act, outputs, mask)
        logp_ref = self._compute_logprobs(self.ref, outputs, mask)
        logp_diff = mask[:, 1:] * (logp_act - logp_ref)
        return outputs, {"kld": logp_diff[tokens.shape[1] - 1 :].sum(-1).mean()}


def get_gradient_slice(grads: ArrayTree, is_jacobian: bool = False) -> Array:
    last_layer_idx = max(map(int, grads["act"]["transformer"]["h"]))
    last_layer_grads = grads["act"]["transformer"]["h"][str(last_layer_idx)]

    arrays = [
        grads["act"]["transformer"]["ln_f"]["scale"],
        last_layer_grads["ln_1"]["scale"],
        last_layer_grads["ln_2"]["scale"],
        last_layer_grads["attn"]["c_attn"]["kernel"],
        last_layer_grads["attn"]["c_proj"]["kernel"],
        last_layer_grads["mlp"]["c_fc"]["kernel"],
        last_layer_grads["mlp"]["c_proj"]["kernel"],
    ]
    flatten_arrays = [
        array.reshape((array.shape[0], -1) if is_jacobian else (-1,))
        for array in arrays
    ]
    return jnp.concatenate(flatten_arrays, axis=-1)


@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    def jacobian_fn(params: ArrayTree) -> ArrayTree:
        return state.apply_fn({"params": params}, *batch)["loss"]

    def lambda_optimize_fn(logits: Array) -> Array:
        grads = (nn.softmax(logits) + state.lamreg * state.lampref) @ jacobian
        return 0.5 * jnp.square(grads).sum()

    # Compute task-wise gradients (a.k.a jacobian matrix) and average them across the
    # devices using `jax.lax.pmean` since this function is wrapped by `jax.pmap`.
    jacobian = jax.jacrev(jacobian_fn)(state.params)
    jacobian = get_gradient_slice(jacobian, is_jacobian=True)
    jacobian = jacobian.reshape(jacobian.shape[0], -1)
    jacobian = jax.lax.pmean(jacobian, axis_name="batch")

    # Apply EMA to the jacobian buffer to estimate global expectation of the gradients.
    # Note that the actual jacobian will be corrected by momentum. Note also that many
    # implementations normalize gradients and consider directions only.
    jacbuf = state.jacmom * state.jacbuf + (1 - state.jacmom) * jacobian
    jacobian = jacbuf / (1 - state.jacmom ** (state.step + 1))
    jacobian = jacobian / jnp.linalg.norm(jacobian, axis=-1, keepdims=True).mean()
    # jacobian = jacobian / jnp.linalg.norm(jacobian, axis=-1, keepdims=True)

    # Update the logits of the lambda vector by using manual SGD. The jacobian is
    # already be debiased by EMA and we use a single loop instead of multiple descent
    # steps for the lambda logits. Note that we use softmax to the logits so that we
    # remove the probability simplex constraint.
    if state.lambuf is not None:
        lamgrads = jax.grad(lambda_optimize_fn)(state.lambuf)
        lambuf = state.lambuf - state.lammom * lamgrads

        weights = nn.softmax(lambuf) + state.lamreg * state.lampref
        weights = weights / (1 + state.lamreg)
    else:
        lambuf, weights = None, jnp.ones(jacobian.shape[0]) / jacobian.shape[0]

    def weighted_loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch)
        metrics["loss"] = weights @ metrics["loss"]
        return metrics["loss"], metrics

    metrics, grads = jax.value_and_grad(weighted_loss_fn, has_aux=True)(state.params)
    metrics, grads = jax.lax.pmean((metrics[1], grads), axis_name="batch")

    metrics |= {f"weight{i}": j for i, j in enumerate(weights)}
    # metrics |= {f"logit{i}": j for i, j in enumerate(lambuf)}
    # metrics |= {f"grad{i}": j for i, j in enumerate(lamgrads)}
    state = state.apply_gradients(grads=grads, jacbuf=jacbuf, lambuf=lambuf)
    return state, metrics | state.opt_state.hyperparams


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,))
def generation_step(
    state: TrainState, tokens: Array, sample_rng: PRNGKey, max_length: int
) -> tuple[Array, ArrayTree]:
    outputs, metrics = state.apply_fn(
        {"params": state.params}, tokens, sample_rng, max_length, method="generate"
    )
    return outputs, jax.lax.pmean(metrics, axis_name="batch")


def create_train_state(
    args: argparse.Namespace, model: FlaxPreTrainedModel, steps_per_epoch: int
) -> TrainState:
    module = TrainModule(
        act=FlaxGPT2LMHeadModule(model.config),
        ref=FlaxGPT2LMHeadModule(model.config),
        penalty_beta=args.penalty_beta,
    )
    params = {"act": model.params, "ref": jax.tree.map(jnp.copy, model.params)}

    jacbuf = get_gradient_slice(params, is_jacobian=False)
    jacbuf = jnp.zeros((len(args.target_columns), jacbuf.size))
    lambuf = jnp.zeros(len(args.target_columns))

    lampref = np.array([float(x.split(":")[2]) for x in args.target_columns])
    lampref = lampref / (lampref.sum() + 1e-10)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
        learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            learning_rate=learning_rate,
            b1=args.adam_b1,
            b2=args.adam_b2,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            mask=partial(jax.tree.map, lambda x: x.ndim > 1),
        )
        if args.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return optax.multi_transform(
            {"act": tx, "ref": optax.set_to_zero()},
            partial(jax.tree_util.tree_map_with_path, lambda path, _: path[0].key),
        )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=args.learning_rate,
        decay_steps=(total_steps := args.epochs * steps_per_epoch),
        warmup_steps=int(args.warmup_ratio * total_steps),
        end_value=0,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        jacbuf=jacbuf,
        jacmom=args.jacmom,
        lambuf=lambuf if args.use_moco else None,
        lammom=args.lammom,
        lamreg=args.lamreg,
        lampref=lampref.astype(np.float32),
    )
