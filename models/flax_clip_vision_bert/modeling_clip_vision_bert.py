from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxSequenceClassifierOutput
)
from transformers.models.bert.modeling_flax_bert import (
    FlaxBertEncoder,
    FlaxBertOnlyMLMHead,
    FlaxBertPooler,
    FlaxPreTrainedModel,
)
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModule

from .configuration_clip_vision_bert import CLIPVisionBertConfig


class FlaxCLIPVisionBertEmbeddings(nn.Module):

    config: CLIPVisionBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        bert_config = self.config.bert_config
        clip_vision_config = self.config.clip_vision_config

        self.word_embeddings = nn.Embed(
            bert_config.vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=bert_config.initializer_range
            ),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            bert_config.max_position_embeddings,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=bert_config.initializer_range
            ),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            bert_config.type_vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=bert_config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.clip_vision_module = FlaxCLIPVisionModule(
            clip_vision_config, dtype=self.dtype
        )
        self.visual_projection = nn.Dense(
            bert_config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                bert_config.initializer_range, self.dtype
            ),
        )

        self.visual_position_embeddings = nn.Embed(
            bert_config.max_position_embeddings,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=bert_config.initializer_range
            ),
            dtype=self.dtype,
        )
        self.visual_token_type_embeddings = nn.Embed(
            bert_config.type_vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=bert_config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.LayerNorm = nn.LayerNorm(
            epsilon=bert_config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=bert_config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        pixel_values,
        visual_token_type_ids,
        visual_position_ids,
        deterministic: bool = True,
    ):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        word_embeddings = inputs_embeds + token_type_embeddings + position_embeds

        # Visual Embed
        visual_inputs_embeds = self.clip_vision_module(pixel_values=pixel_values)[0]
        visual_inputs_embeds = self.visual_projection(visual_inputs_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(
            visual_token_type_ids.astype("i4")
        )
        visual_position_embeds = self.visual_position_embeddings(
            visual_position_ids.astype("i4")
        )

        # Sum all visual embeddings
        visual_embeddings = (
            visual_inputs_embeds + visual_token_type_embeddings + visual_position_embeds
        )

        # Concat
        hidden_states = jnp.concatenate((word_embeddings, visual_embeddings), axis=1)

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxCLIPVisionBertModule(nn.Module):
    config: CLIPVisionBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxCLIPVisionBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(self.config.bert_config, dtype=self.dtype)
        self.pooler = FlaxBertPooler(self.config.bert_config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        pixel_values,
        visual_attention_mask,
        visual_token_type_ids,
        visual_position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(
            input_ids,
            token_type_ids,
            position_ids,
            pixel_values,
            visual_token_type_ids,
            visual_position_ids,
            deterministic=deterministic,
        )

        combined_attention_mask = jnp.concatenate(
            (attention_mask, visual_attention_mask), axis=1
        )

        outputs = self.encoder(
            hidden_states,
            combined_attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxCLIPVisionBertModel(FlaxPreTrainedModel):
    config_class = CLIPVisionBertConfig
    module_class = FlaxCLIPVisionBertModule

    def __init__(
        self,
        config: CLIPVisionBertConfig,
        input_shape: Tuple = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):

        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.clip_vision_config.image_size,
                    config.clip_vision_config.image_size,
                    3,
                ),
                (
                    1,
                    (
                        config.clip_vision_config.image_size
                        // config.clip_vision_config.patch_size
                    )
                    ** 2
                    + 1,
                ),
            )

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config, module, input_shape=input_shape, seed=seed, dtype=dtype
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        textual_input_shape = input_shape[0]
        input_ids = jnp.zeros(textual_input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), textual_input_shape
        )
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])
        visual_attention_mask = jnp.ones(input_shape[2])
        visual_token_type_ids = jnp.ones(input_shape[2])
        visual_position_ids = jnp.broadcast_to(
            jnp.zeros(jnp.atleast_2d(visual_token_type_ids).shape[-1]), input_shape[2]
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            pixel_values,
            visual_attention_mask,
            visual_token_type_ids,
            visual_position_ids,
            return_dict=False,
        )["params"]

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pixel_values=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        visual_position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.bert_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.bert_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.bert_config.return_dict
        )

        # pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1)) # Don't need this for torch permuted input

        visual_sequence_length = (
            pixel_values.shape[0],
            (
                self.config.clip_vision_config.image_size
                // self.config.clip_vision_config.patch_size
            )
            ** 2
            + 1,
        )
        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if visual_token_type_ids is None:
            visual_token_type_ids = jnp.ones(visual_sequence_length)

        if visual_position_ids is None:
            visual_position_ids = jnp.broadcast_to(
                jnp.atleast_2d(visual_token_type_ids).shape[-1], visual_sequence_length
            )

        if visual_attention_mask is None:
            visual_attention_mask = jnp.ones(visual_sequence_length)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(visual_attention_mask, dtype="i4"),
            jnp.array(visual_token_type_ids, dtype="i4"),
            jnp.array(visual_position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    @classmethod
    def from_bert_clip_vision_pretrained(
        cls,
        bert_model_name_or_path: str = None,
        clip_vision_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:

        kwargs_bert = {
            argument[len("bert_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        kwargs_clip_vision = {
            argument[len("clip_vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_bert.keys():
            del kwargs["bert_" + key]
        for key in kwargs_clip_vision.keys():
            del kwargs["clip_vision_" + key]

        # Load and initialize the text and vision model
        bert_model = kwargs_bert.pop("model", None)
        if bert_model is None:
            assert (
                bert_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `bert_model_name_or_path` has to be defined"
            from transformers import FlaxBertModel

            if "config" not in kwargs_bert:
                from transformers import BertConfig

                bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
                kwargs_bert["config"] = bert_config

            bert_model = FlaxBertModel.from_pretrained(
                bert_model_name_or_path, *model_args, from_pt=True, **kwargs_bert
            )

        clip_vision_model = kwargs_clip_vision.pop("model", None)
        if clip_vision_model is None:
            assert (
                clip_vision_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `clip_vision_model_name_or_path` has to be defined"
            from transformers import FlaxCLIPVisionModel

            if "config" not in kwargs_clip_vision:
                from transformers import CLIPVisionConfig

                clip_vision_config = CLIPVisionConfig.from_pretrained(
                    clip_vision_model_name_or_path
                )
                kwargs_clip_vision["config"] = clip_vision_config

            clip_vision_model = FlaxCLIPVisionModel.from_pretrained(
                clip_vision_model_name_or_path, *model_args, **kwargs_clip_vision
            )

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = CLIPVisionBertConfig.from_bert_clip_vision_configs(
            bert_model.config, clip_vision_model.config, **kwargs
        )

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        for key in model.params.keys():
            if key != "embeddings":
                model.params[key] = bert_model.params[key]
            else:
                model.params["embeddings"][
                    "clip_vision_module"
                ] = clip_vision_model.params
                for sub_key in bert_model.params[key]:
                    model.params[key][sub_key] = bert_model.params[key][sub_key]

        return model


# flax_model = FlaxCLIPVisionBertModel.from_bert_clip_vision_pretrained('bert-base-uncased', 'openai/clip-vit-base-patch32', seed=0, dtype=jnp.float32)
# outputs = flax_model(input_ids, attention_mask,token_type_ids, position_ids, pixel_values, visual_attention_mask, visual_token_type_ids, visual_position_ids, output_hidden_states=True)


class FlaxCLIPVisionBertForMaskedLMModule(nn.Module):
    config: CLIPVisionBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxCLIPVisionBertModule(
            config=self.config, add_pooling_layer=False, dtype=self.dtype
        )
        self.cls = FlaxBertOnlyMLMHead(config=self.config.bert_config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        pixel_values,
        visual_attention_mask,
        visual_token_type_ids,
        visual_position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        # Model
        outputs = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            pixel_values,
            visual_attention_mask,
            visual_token_type_ids,
            visual_position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.bert_config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["embeddings"][
                "word_embeddings"
            ]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxCLIPVisionBertForMaskedLM(FlaxPreTrainedModel):
    config_class = CLIPVisionBertConfig
    module_class = FlaxCLIPVisionBertForMaskedLMModule

    def __init__(
        self,
        config: CLIPVisionBertConfig,
        input_shape: Tuple = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):

        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.clip_vision_config.image_size,
                    config.clip_vision_config.image_size,
                    3,
                ),
                (
                    1,
                    (
                        config.clip_vision_config.image_size
                        // config.clip_vision_config.patch_size
                    )
                    ** 2
                    + 1,
                ),
            )

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config, module, input_shape=input_shape, seed=seed, dtype=dtype
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        textual_input_shape = input_shape[0]
        input_ids = jnp.zeros(textual_input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), textual_input_shape
        )
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])
        visual_attention_mask = jnp.ones(input_shape[2])
        visual_token_type_ids = jnp.ones(input_shape[2])
        visual_position_ids = jnp.broadcast_to(
            jnp.zeros(jnp.atleast_2d(visual_token_type_ids).shape[-1]), input_shape[2]
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            pixel_values,
            visual_attention_mask,
            visual_token_type_ids,
            visual_position_ids,
            return_dict=False,
        )["params"]

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pixel_values=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        visual_position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.bert_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.bert_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.bert_config.return_dict
        )

        # pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        visual_sequence_length = (
            pixel_values.shape[0],
            (
                self.config.clip_vision_config.image_size
                // self.config.clip_vision_config.patch_size
            )
            ** 2
            + 1,
        )

        if visual_token_type_ids is None:
            visual_token_type_ids = jnp.ones(visual_sequence_length)

        if visual_position_ids is None:
            visual_position_ids = jnp.broadcast_to(
                jnp.atleast_2d(jnp.ones(visual_sequence_length)).shape[-1],
                (visual_sequence_length),
            )

        if visual_attention_mask is None:
            visual_attention_mask = jnp.ones(visual_sequence_length)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(visual_attention_mask, dtype="i4"),
            jnp.array(visual_token_type_ids, dtype="i4"),
            jnp.array(visual_position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        # kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_clip_vision_bert_pretrained(
        cls,
        clip_vision_model_name_or_path: str = None,
        bert_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:

        kwargs_bert = {
            argument[len("bert_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        kwargs_clip_vision = {
            argument[len("clip_vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_bert.keys():
            del kwargs["bert_" + key]
        for key in kwargs_clip_vision.keys():
            del kwargs["clip_vision_" + key]

        # Load and initialize the text and vision model
        bert_model = kwargs_bert.pop("model", None)
        if bert_model is None:
            assert (
                bert_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `bert_model_name_or_path` has to be defined"
            from transformers import FlaxBertForMaskedLM

            if "config" not in kwargs_bert:
                from transformers import BertConfig

                bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
                kwargs_bert["config"] = bert_config

            bert_model = FlaxBertForMaskedLM.from_pretrained(
                bert_model_name_or_path, *model_args, from_pt=True, **kwargs_bert
            )

        clip_vision_model = kwargs_clip_vision.pop("model", None)
        if clip_vision_model is None:
            assert (
                clip_vision_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `clip_vision_model_name_or_path` has to be defined"
            from transformers import FlaxCLIPVisionModel

            if "config" not in kwargs_clip_vision:
                from transformers import CLIPVisionConfig

                clip_vision_config = CLIPVisionConfig.from_pretrained(
                    clip_vision_model_name_or_path
                )
                kwargs_clip_vision["config"] = clip_vision_config

            clip_vision_model = FlaxCLIPVisionModel.from_pretrained(
                clip_vision_model_name_or_path, *model_args, **kwargs_clip_vision
            )

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = CLIPVisionBertConfig.from_clip_vision_bert_configs(
            clip_vision_model.config, bert_model.config, **kwargs
        )

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        model.params["cls"] = bert_model.params["cls"]
        for key in model.params["model"].keys():
            if key != "embeddings":
                model.params["model"][key] = bert_model.params["bert"][key]
            else:
                model.params["model"]["embeddings"][
                    "clip_vision_module"
                ] = clip_vision_model.params
                for sub_key in bert_model.params["bert"][key]:
                    model.params["model"][key][sub_key] = bert_model.params["bert"][
                        key
                    ][sub_key]

        return model


class FlaxCLIPVisionBertForSequenceClassificationModule(nn.Module):
    config: CLIPVisionBertConfig
    dtype: jnp.dtype = jnp.float32
    num_labels:int = 2

    def setup(self):
        self.model = FlaxCLIPVisionBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.bert_config.hidden_dropout_prob)
        self.classifier = nn.Dense(
            self.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        pixel_values,
        visual_attention_mask,
        visual_token_type_ids, 
        visual_position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                pixel_values,
                visual_attention_mask,
                visual_token_type_ids, 
                visual_position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FlaxCLIPVisionBertForSequenceClassification(FlaxPreTrainedModel):
    config_class = CLIPVisionBertConfig
    module_class = FlaxCLIPVisionBertForSequenceClassificationModule    

    def __init__(
        self,
        config: CLIPVisionBertConfig,
        input_shape: Tuple = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):

        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.clip_vision_config.image_size,
                    config.clip_vision_config.image_size,
                    3,
                ),
                (
                    1,
                    (
                        config.clip_vision_config.image_size
                        // config.clip_vision_config.patch_size
                    )
                    ** 2
                    + 1,
                ),
            )

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config, module, input_shape=input_shape, seed=seed, dtype=dtype
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        textual_input_shape = input_shape[0]
        input_ids = jnp.zeros(textual_input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), textual_input_shape
        )
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])
        visual_attention_mask = jnp.ones(input_shape[2])
        visual_token_type_ids = jnp.ones(input_shape[2])
        visual_position_ids = jnp.broadcast_to(
            jnp.zeros(jnp.atleast_2d(visual_token_type_ids).shape[-1]), input_shape[2]
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            pixel_values,
            visual_attention_mask,
            visual_token_type_ids,
            visual_position_ids,
            return_dict=False,
        )["params"]

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pixel_values=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        visual_position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.bert_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.bert_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.bert_config.return_dict
        )

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        visual_sequence_length = (
            pixel_values.shape[0],
            (
                self.config.clip_vision_config.image_size
                // self.config.clip_vision_config.patch_size
            )
            ** 2
            + 1,
        )

        if visual_token_type_ids is None:
            visual_token_type_ids = jnp.ones(visual_sequence_length)

        if visual_position_ids is None:
            visual_position_ids = jnp.broadcast_to(
                jnp.atleast_2d(jnp.ones(visual_sequence_length)).shape[-1],
                (visual_sequence_length),
            )

        if visual_attention_mask is None:
            visual_attention_mask = jnp.ones(visual_sequence_length)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(visual_attention_mask, dtype="i4"),
            jnp.array(visual_token_type_ids, dtype="i4"),
            jnp.array(visual_position_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        # kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_clip_vision_bert_pretrained(
        cls,
        clip_vision_model_name_or_path: str = None,
        bert_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:

        kwargs_bert = {
            argument[len("bert_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("bert_")
        }

        kwargs_clip_vision = {
            argument[len("clip_vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("clip_vision_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_bert.keys():
            del kwargs["bert_" + key]
        for key in kwargs_clip_vision.keys():
            del kwargs["clip_vision_" + key]

        # Load and initialize the text and vision model
        bert_model = kwargs_bert.pop("model", None)
        if bert_model is None:
            assert (
                bert_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `bert_model_name_or_path` has to be defined"
            from transformers import FlaxBertForSequenceClassification

            if "config" not in kwargs_bert:
                from transformers import BertConfig

                bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
                kwargs_bert["config"] = bert_config

            bert_model = FlaxBertForSequenceClassification.from_pretrained(
                bert_model_name_or_path, *model_args, from_pt=True, **kwargs_bert
            )

        clip_vision_model = kwargs_clip_vision.pop("model", None)
        if clip_vision_model is None:
            assert (
                clip_vision_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `clip_vision_model_name_or_path` has to be defined"
            from transformers import FlaxCLIPVisionModel

            if "config" not in kwargs_clip_vision:
                from transformers import CLIPVisionConfig

                clip_vision_config = CLIPVisionConfig.from_pretrained(
                    clip_vision_model_name_or_path
                )
                kwargs_clip_vision["config"] = clip_vision_config

            clip_vision_model = FlaxCLIPVisionModel.from_pretrained(
                clip_vision_model_name_or_path, *model_args, **kwargs_clip_vision
            )

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = CLIPVisionBertConfig.from_clip_vision_bert_configs(
            clip_vision_model.config, bert_model.config, **kwargs
        )

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        #model.params["classifier"] = bert_model.params["classifier"]
        for key in model.params["model"].keys():
            if key != "embeddings":
                model.params["model"][key] = bert_model.params["bert"][key]
            else:
                model.params["model"]["embeddings"][
                    "clip_vision_module"
                ] = clip_vision_model.params
                for sub_key in bert_model.params["bert"][key]:
                    model.params["model"][key][sub_key] = bert_model.params["bert"][
                        key
                    ][sub_key]

        return model