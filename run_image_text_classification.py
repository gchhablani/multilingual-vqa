#!/usr/bin/env python3
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""Script to run Image-Text Masked LM"""
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import torch
from flax import jax_utils, struct, traverse_util
from itertools import chain
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from jax.random import PRNGKey
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from transformers.file_utils import PushToHubMixin

from models.flax_clip_vision_bert.configuration_clip_vision_bert import (
    CLIPVisionBertConfig,
)
from models.flax_clip_vision_bert.modeling_clip_vision_bert import (
    FlaxCLIPVisionBertForSequenceClassification
)
from datasets import load_metric
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Array = Any
# Args
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    clip_vision_name_or_path: Optional[str] = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "The bert model checkpoint for weights initialization."},
    )

    bert_name_or_path: Optional[str] = field(
        default="bert-base-multilingual-uncased",
        metadata={"help": "The bert model checkpoint for weights initialization."},
    )

    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The pretrained checkpoint for weights initialization. Takes highest precedence."},
    )

    bert_tokenizer_name: Optional[str] = field(
        default="bert-base-multilingual-uncased",
        metadata={
            "help": "Pretrained BERT tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    # save_optimizer: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "Whether to save optimizer state."
    #     },
    # )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default="./images/",
        metadata={"help": "The data directory containing input files."},
    )
    train_file: Optional[str] = field(
        default="train.tsv",
        metadata={"help": "The input training data file (a tsv file)."},
    )
    validation_file: Optional[str] = field(
        default="test.tsv",
        metadata={"help": "An optional input evaluation data file (a tsv file)."},
    )
    # image_size: Optional[int] = field(default=224, metadata={"help": " The size (resolution) of each image."}) # TODO: Check

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need both training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "tsv", "`train_file` should be a tsv."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "tsv", "`validation_file` should be a tsv."


# Transform

# We use torchvision for faster image pre-processing.
# We need to ensure faster processing speed as it can become a bottleneck on TPU
class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


# ImageTextDataset
class ImageTextDataset(VisionDataset):

    def __init__(
        self,
        root: str,
        file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        max_samples: int = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        examples = pd.read_csv(file_path, sep="\t")

        image_paths = []
        questions = []
        labels = []
        for idx, img_file in enumerate(examples["image_file"].values):
            if os.path.exists(os.path.join(self.root, img_file)):
                image_paths.append(img_file)
                questions.append(examples["question"].values[idx])
                labels.append(examples["answer_label"].values[idx])

        if max_samples is None:
            max_samples = len(questions)
        self.image_paths = image_paths[:max_samples]
        self.questions = questions[:max_samples]
        self.labels = labels[:max_samples]

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(os.path.join(self.root, path), mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.questions[idx]

    def _load_label(self, idx):
        return self.labels[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        question = str(self._load_target(index))
        label = self._load_label(index)

        if self.transforms is not None:
            image, question = self.transforms(image, question)

        return image, question, label

    def __len__(self) -> int:
        return len(self.questions)


# Data Collator


@flax.struct.dataclass
class FlaxDataCollatorForImageTextSequenceClassification:

    tokenizer: PreTrainedTokenizerBase
    max_length:int = 128
    vision_sequence_length:int = 50
    
    def __call__(self, examples) -> Dict[str, np.ndarray]:

        pixel_values = (
            torch.stack([example[0] for example in examples])
            .permute(0, 2, 3, 1)
            .numpy()
        )
        questions = [example[1] for example in examples]
        labels = np.array([example[2] for example in examples])

        batch = self.tokenizer(
            questions,
            padding="max_length",
            max_length=self.max_length - self.vision_sequence_length,
            return_tensors=TensorType.NUMPY,
            truncation=True,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "labels": labels,
        }


def create_train_state(
    model: FlaxCLIPVisionBertForSequenceClassification,
    learning_rate_fn: Callable[[int], float],
    is_regression: bool,
    num_labels: int,
    weight_decay: float,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """

        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale"))
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
        mask=decay_mask_fn,
    )

    if is_regression:

        def mse_loss(logits, labels):
            return jnp.mean((logits[..., 0] - labels) ** 2)

        return TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=tx,
            logits_fn=lambda logits: logits[..., 0],
            loss_fn=mse_loss,
        )
    else:  # Classification.

        def cross_entropy_loss(logits, labels):
            xentropy = optax.softmax_cross_entropy(
                logits, onehot(labels, num_classes=num_labels)
            )
            return jnp.mean(xentropy)

        return TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=tx,
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=cross_entropy_loss,
        )

# Helper Methods


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


# utils
def mb_item(x):
    return x.item() if hasattr(x, "item") else x


# checkpoint functions
def save_model_checkpoint(
    model,
    save_dir,
    state,
    logger,
    organization,
    with_opt: bool = False,
    push_to_hub: bool = False,
    overwrite=False,
    **kwargs,
):
    state = jax_utils.unreplicate(state)
    logger.info(f"Saving Checkpoint in {save_dir}")
    ckpt_save_dir = f"{save_dir}/ckpt-{mb_item(state.step)-1}"
    if os.path.exists(ckpt_save_dir) and not overwrite:
        logger.info("checkpoint exists, skipping overwrite")
    else:
        model.save_pretrained(
            ckpt_save_dir, params=state.params, push_to_hub=False, **kwargs
        )
        if with_opt:
            with open(os.path.join(ckpt_save_dir, "opt_state.msgpack"), "wb") as f:
                f.write(to_bytes(state.opt_state))
            with open(os.path.join(ckpt_save_dir, "training_state.json"), "w") as f:
                json.dump({"step": state.step.item()}, f)

        logger.info("checkpoint saved")

        if push_to_hub:
            repo_name = Path(save_dir).name
            repo_url = PushToHubMixin._get_repo_url_from_name(
                repo_name, organization=organization, private=False, use_auth_token=True
            )
            repo = PushToHubMixin._create_or_get_repo(
                save_dir,
                repo_url=repo_url,
                organization=organization,
                use_auth_token=True,
            )
            commit_message = f"Saving weights and logs at step {mb_item(state.step)-1}"
            url = PushToHubMixin._push_to_hub(repo=repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")


def restore_model_checkpoint(save_dir, state, logger):
    logger.info(f"Restoring checkpoint from {save_dir}.")
    with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())

    with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())

    with open(os.path.join(save_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]

    logger.info("checkpoint restored")
    # return state.replace(step=step, params=params, opt_state=opt_state), step
    return params, opt_state, step


def rotate_checkpoints(ckpt_dir: str, save_total_limit: int, logger):
    "Removes older checkpoints so that `save_total_limit` checkpoints are kept"
    # TODO: what to remove is decided using step number only, we might want to improve that
    ckpts = [str(x) for x in Path(ckpt_dir).glob("ckpt-*")]
    # sort checkpoints by step
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    ckpts_to_delete = ckpts_sorted[:-save_total_limit]
    for ckpt in ckpts_to_delete:
        logger.info(
            f"Deleting older checkpoint [{ckpt}] due to save_total_limit ({save_total_limit})"
        )
        shutil.rmtree(ckpt)


# Main
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # TODO: Need to fix this
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Model
    if training_args.resume_from_checkpoint is None and model_args.pretrained_checkpoint is None:
        model = FlaxCLIPVisionBertForSequenceClassification.from_clip_vision_bert_pretrained(
            model_args.clip_vision_name_or_path,
            model_args.bert_name_or_path,
            num_labels=3129,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )
    elif model_args.pretrained_checkpoint is not None:
        model = FlaxCLIPVisionBertForSequenceClassification.from_pretrained(
            model_args.pretrained_checkpoint
        )
    else:
        model = FlaxCLIPVisionBertForSequenceClassification.from_pretrained(
            training_args.resume_from_checkpoint
        )

    config = model.config

    # Dataset
    preprocess = Transform(config.clip_vision_config.image_size)
    preprocess = torch.jit.script(preprocess)

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        transform=preprocess,
        max_samples=data_args.max_train_samples,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        transform=preprocess,
        max_samples=data_args.max_eval_samples,
    )

    # Tokenizer

    if model_args.bert_tokenizer_name:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_args.bert_tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Data Collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForImageTextSequenceClassification(
        tokenizer=tokenizer
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # State
    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Setup train state
    # state = TrainState.create(
    #     apply_fn=model.__call__,
    #     params=model.params,
    #     tx=optimizer,
    #     dropout_rng=dropout_rng,
    # )

    # Setup train state

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    state = create_train_state(
        model,
        learning_rate_fn,
        is_regression=False,
        num_labels=3129,
        weight_decay=training_args.weight_decay,
    )
    if training_args.resume_from_checkpoint is not None:
        params, opt_state, step = restore_model_checkpoint(
            training_args.resume_from_checkpoint, state, logger
        )
        state = state.replace(
            step=step,
            apply_fn=model.__call__,
            params=params,
            tx=optimizer,
            opt_state=opt_state,
        )

    # Train Step

    # Define gradient update step fn
    def train_step(
        state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey
    ) -> Tuple[train_state.TrainState, float]:
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        targets = batch.pop("labels")

        def loss_fn(params):
            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            loss = state.loss_fn(logits, targets)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": learning_rate_fn(state.step)},
            axis_name="batch",
        )
        return new_state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

    # Eval Step
    # Define eval fn
    def eval_step(state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        return state.logits_fn(logits)

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    # Train Loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")
    if training_args.resume_from_checkpoint is not None:
        previous_step = int(jax_utils.unreplicate(state.step))
        epoch_start_point = math.ceil(
            (previous_step * train_batch_size) / len(train_dataset)
        )
    else:
        epoch_start_point = 0

    break_all = False
    train_time = 0
    epochs = tqdm(
        range(epoch_start_point, num_epochs),
        desc=f"Epoch:  ({epoch_start_point+1}/{num_epochs})",
        position=0,
    )
    metric = load_metric("accuracy")
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(train_dataset)

        epochs.desc = f"Epoch:  ({epoch+1}/{num_epochs})"

        train_step_progress_bar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch+1}: ", position=0, leave=False
        )
        # Gather the indexes for creating the batch and do a training step

        for step, batch in enumerate(train_loader):
            # print(batch.keys())
            # print(batch['labels'].shape)
            batch = shard(batch)
            state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            cur_step = epoch * (num_train_samples // train_batch_size) + step + 1

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(
                        summary_writer, train_metrics, train_time, cur_step
                    )

                epochs.write(
                    f"Log at Step: {cur_step} (Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
                )

                train_metrics = (
                    []
                )  # TODO: Check why is this being done? WHat is this needed for?

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(eval_dataset)
                # eval_samples_idx = jnp.arange(num_eval_samples)
                # eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)
                eval_metrics = []
                eval_steps = len(eval_dataset) // eval_batch_size
                eval_step_progress_bar = tqdm(
                    total=eval_steps, desc="Evaluating: ", position=2, leave=False
                )
                for batch in eval_loader:

                    # Model forward
                    batch = shard(batch)
                    labels = batch.pop("labels")
                    predictions = p_eval_step(state, batch)
                    metric.add_batch(predictions=chain(*predictions), references=chain(*labels))
                    eval_step_progress_bar.update(1)

                eval_metric = metric.compute()
                # Update progress bar
                epochs.write(
                    f"Eval at Step: {cur_step} (Accuracy: {eval_metric})"
                )

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metric, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    # params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    # model.save_pretrained(
                    #     training_args.output_dir,
                    #     params=params,
                    #     push_to_hub=training_args.push_to_hub,
                    #     commit_message=f"Saving weights and logs of step {cur_step}",
                    # )
                    save_model_checkpoint(
                        model,
                        training_args.output_dir,
                        state,
                        logger,
                        training_args.push_to_hub_organization,
                        with_opt=True,
                        push_to_hub=training_args.push_to_hub,
                        overwrite=True,
                    )
                    # if model_args.save_optimizer:
                    #     # this saves full state including optimizer
                    #     save_checkpoint(training_args.output_dir, state, state.step, keep=training_args.save_total_limit, overwrite=True)
                    if training_args.save_total_limit is not None:
                        rotate_checkpoints(
                            training_args.output_dir,
                            training_args.save_total_limit,
                            logger,
                        )

            if cur_step == total_train_steps:
                break_all = True
                break
        train_step_progress_bar.close()
        epochs.update(1)
        if break_all:
            break
    # save model after training is over
    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
    model.save_pretrained(
        training_args.output_dir,
        params=params,
        push_to_hub=training_args.push_to_hub,
        commit_message="Add final model",
    )


if __name__ == "__main__":
    main()
