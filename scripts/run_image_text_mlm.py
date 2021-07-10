#!/usr/bin/env python
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
import logging
import os
import time
import pandas as pd
from dataclasses import dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import numpy as np
from tqdm import tqdm


from torchvision.transforms.functional import InterpolationMode
import flax
from flax.training.common_utils import get_metrics, shard, shard_prng_key
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    BertTokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)

from models.flax_clip_vision_bert.configuration_clip_vision_bert import CLIPVisionBertConfig
from models.flax_clip_vision_bert.modeling_clip_vision_bert import FlaxCLIPVisionBertForMaskedLM


# Args
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    clip_vision_name_or_path: Optional[str] = field(
        default='openai/clip-vit-base-patch32',
        metadata={
            "help": "The bert model checkpoint for weights initialization."
        },
    )

    bert_name_or_path: Optional[str] = field(
        default='bert-base-multilingual-uncased',
        metadata={
            "help": "The bert model checkpoint for weights initialization."
        },
    )

    bert_tokenizer_name: Optional[str] = field(
        default='bert-base-multilingual-uncased', metadata={"help": "Pretrained BERT tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
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

    data_dir: Optional[str] = field(default='./images/', metadata={"help": "The data directory containing input files."})
    train_file: Optional[str] = field(
        default='train.tsv', metadata={"help": "The input training data file (a tsv file)."}
    )
    validation_file: Optional[str] = field(
        default='test.tsv',
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
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    # pad_to_max_length: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "Whether to pad all samples to `max_seq_length`. "
    #         "If False, will pad the samples dynamically when batching to the maximum length in the batch."
    #     },
    # ) # TODO: CHECK

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
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x



# ImageTextDataset

class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        examples = pd.read_csv(file_path, sep='\t')

        print(examples.columns)
        self.image_paths = examples["image_file"].values
        self.captions = examples["caption"].values
        

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(os.path.join(self.root,path), mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)

# Data Collator

@flax.struct.dataclass
class FlaxDataCollatorForImageLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples) -> Dict[str, np.ndarray]:

        pixel_values = torch.stack([example[0] for example in examples]).permute(0, 2, 3, 1).numpy()
        captions = [example[1] for example in examples]
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer(captions, return_special_tokens_mask=True, return_tensors=TensorType.NUMPY) # TODO: Check if truncation is needed.

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None) # TODO: Check how to get `special_tokens-mask`

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        batch["pixel_values"]=pixel_values
        return batch

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# Train State

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


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
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

# Main

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

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

    model = FlaxCLIPVisionBertForMaskedLM.from_clip_vision_bert_pretrained(
        model_args.clip_vision_name_or_path,
        model_args.bert_name_or_path,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
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
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        transform=preprocess,
    )

    # Tokenizer

    if model_args.bert_tokenizer_name:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_args.bert_tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Data Collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForImageLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)


    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
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

    # Optimizer

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBERT-like models.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # # create adam optimizer
    # if training_args.adafactor:
    #     # We use the default parameters here to initialize adafactor,
    #     # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
    #     optimizer = optax.adafactor(
    #         learning_rate=linear_decay_lr_schedule_fn,
    #     )
    # else:
    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # State
    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

    # Train Step

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

            # compute loss, ignore padded input tokens
            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

            # take average
            loss = loss.sum() / label_mask.sum()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Eval Step

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)


    # Train Loop

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(train_dataset)

        # Gather the indexes for creating the batch and do a training step
        for step, batch in enumerate(tqdm(train_loader, desc="Training...", position=1)):
            batch = shard(batch.data)
            state, train_metric, dropout_rng = p_train_step(state, model, dropout_rng)
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
                )

                train_metrics = [] # TODO: Check why is this being done? WHat is this needed for?

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(eval_dataset)
                # eval_samples_idx = jnp.arange(num_eval_samples)
                # eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

                eval_metrics = []
                for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating ...", position=2)):

                    # Model forward
                    batch = shard(batch.data)
                    metrics = p_eval_step(state.params, batch)
                    eval_metrics.append(metrics)

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
                eval_normalizer = eval_metrics.pop("normalizer")
                eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

                # Update progress bar
                epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(
                        training_args.output_dir,
                        params=params,
                        push_to_hub=training_args.push_to_hub,
                        commit_message=f"Saving weights and logs of step {cur_step}",
                    )

if __name__ == "__main__":
    main()