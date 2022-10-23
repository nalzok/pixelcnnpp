# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PixelCNN++ example."""

# See issue #620.
# pytype: disable=wrong-keyword-args

from functools import partial
import datetime

from absl import logging
from flax import jax_utils
from flax.training import train_state
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

import input_pipeline
import pixelcnn


def get_summary_writers(workdir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = workdir + "/log/" + current_time
    train_log_dir = log_dir + "/train"
    eval_log_dir = log_dir + "/eval"
    train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
    eval_summary_writer = tensorboard.SummaryWriter(eval_log_dir)
    return train_summary_writer, eval_summary_writer


def model(config: ml_collections.ConfigDict, **kwargs):
    return pixelcnn.PixelCNNPP(
        depth=config.n_resnet,
        features=config.n_feature,
        logistic_components=config.n_logistic_mix,
        dropout_p=config.dropout_rate,
        **kwargs
    )


def neg_log_likelihood_loss(nn_out, images):
    # The log-likelihood in bits per pixel-channel
    means, inv_scales, logit_weights = pixelcnn.conditional_params_from_outputs(
        nn_out, images
    )
    log_likelihoods = pixelcnn.logprob_from_conditional_params(
        images, means, inv_scales, logit_weights
    )
    return -jnp.mean(log_likelihoods) / (jnp.log(2) * np.prod(images.shape[-3:]))


class TrainState(train_state.TrainState):
    polyak_decay: float


def create_train_state(rng, config: ml_collections.ConfigDict, init_batch: jax.numpy.ndarray):
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    cnn = model(config)
    params = cnn.init(
        {"params": init_rng, "dropout": dropout_rng}, init_batch, train=False
    )
    tx = optax.adamw(
        config.learning_rate,
        b1=0.95,
        b2=0.9995,
        weight_decay=1-config.lr_decay
    )
    state = TrainState.create(
            apply_fn=cnn.apply, params=params, tx=tx, polyak_decay=config.polyak_decay)

    return rng, state


@partial(jax.pmap, axis_name="batch")
def train_step(
    state: TrainState,
    batch,
    dropout_rng,
):
    """Perform a single training step."""

    @jax.value_and_grad
    def loss_fn(params):
        """loss function used for training."""
        pcnn_out = state.apply_fn(
            params,
            batch["image"],
            rngs={"dropout": dropout_rng},
            train=True,
        )
        return neg_log_likelihood_loss(pcnn_out, batch["image"])

    params = state.params
    loss, grads = loss_fn(params)
    grads = jax.lax.pmean(grads, "batch")
    state = state.apply_gradients(grads=grads)

    # Compute exponential moving average (aka Polyak decay)
    ema_decay = state.polyak_decay
    params = jax.tree_util.tree_map(
        lambda old, new: ema_decay * old + (1 - ema_decay) * new, params, state.params
    )
    state = state.replace(params=params)

    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return state, metrics


@partial(jax.pmap, axis_name="batch")
def eval_step(state: TrainState, batch):
    images = batch["image"]
    pcnn_out = state.apply_fn(state.params, images, train=False)
    return {"loss": jax.lax.pmean(neg_log_likelihood_loss(pcnn_out, images), "batch")}


def load_and_shard_tf_batch(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    tf.io.gfile.makedirs(workdir)

    batch_size = config.batch_size
    n_devices = jax.device_count()
    if jax.process_count() > 1:
        raise ValueError(
            "PixelCNN++ example should not be run on more than 1 host" " (for now)"
        )
    if batch_size % n_devices > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    train_summary_writer, eval_summary_writer = get_summary_writers(workdir)
    # Load dataset
    data_source = input_pipeline.DataSource(config)
    train_ds = data_source.train_ds
    eval_ds = data_source.eval_ds
    steps_per_epoch = (
        data_source.ds_info.splits["train"].num_examples // config.batch_size
    )
    # Create dataset batch iterators
    train_iter = iter(train_ds)
    num_train_steps = train_ds.cardinality().numpy()
    steps_per_checkpoint = 1000

    # Create the model using data-dependent initialization. Don't shard the init
    # batch.
    assert config.init_batch_size <= batch_size
    init_batch = next(train_iter)["image"]._numpy()[: config.init_batch_size]

    rng = jax.random.PRNGKey(config.seed)
    rng, state = create_train_state(rng, config, init_batch)

    state = checkpoints.restore_checkpoint(workdir, state)
    step_offset = int(state.step)

    state = jax_utils.replicate(state)

    # Gather metrics
    train_metrics = []

    for step, batch in zip(range(step_offset, num_train_steps), train_iter):
        # Load and shard the TF batch
        batch = load_and_shard_tf_batch(batch)

        # Generate a PRNG key that will be rolled into the batch.
        rng, step_rng = jax.random.split(rng)
        sharded_rngs = common_utils.shard_prng_key(step_rng)

        # Train step
        state, metrics = train_step(state, batch, sharded_rngs)
        train_metrics.append(metrics)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            # We've finished an epoch
            train_metrics = common_utils.get_metrics(train_metrics)
            # Get training epoch summary for logging
            train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
            # Send stats to Tensorboard
            for key, vals in train_metrics.items():
                for i, val in enumerate(vals):
                    train_summary_writer.scalar(key, val, step - len(vals) + i + 1)
            # Reset train metrics
            train_metrics = []

            # Evaluation
            eval_metrics = []
            for eval_batch in eval_ds:
                # Load and shard the TF batch
                eval_batch = load_and_shard_tf_batch(eval_batch)
                # Step
                metrics = eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            # Get eval epoch summary for logging
            eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

            # Log epoch summary
            logging.info(
                "Epoch %d: TRAIN loss=%.6f, EVAL loss=%.6f",
                epoch,
                train_summary["loss"],
                eval_summary["loss"],
            )

            eval_summary_writer.scalar("loss", eval_summary["loss"], step)
            train_summary_writer.flush()
            eval_summary_writer.flush()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_train_steps:
            checkpoints.save_checkpoint(workdir, jax_utils.unreplicate(state), step, keep=3)