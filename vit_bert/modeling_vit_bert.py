from transformers.models.bert.modeling_flax_bert import FlaxPreTrainedModel, FlaxBertEncoder, FlaxBertPooler, FlaxBaseModelOutputWithPooling
from transformers.models.vit.modeling_flax_vit import FlaxViTModule
from typing import Tuple, Optional
from flax.core.frozen_dict import FrozenDict
import jax
import flax.linen as nn
from configuration_vit_bert import ViTBertConfig


class FlaxViTBertEmbeddings(nn.Module):

    config: ViTBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        bert_config = self.config.bert_config
        vit_config = self.config.vit_config

        self.word_embeddings = nn.Embed(
            bert_config.vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=bert_config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = nn.Embed(
            bert_config.max_position_embeddings,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=bert_config.initializer_range),
            dtype=self.dtype,
        )
        self.token_type_embeddings = nn.Embed(
            bert_config.type_vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=bert_config.initializer_range),
            dtype=self.dtype,
        )

        self.vit_module = FlaxViTModule(vit_config, dtype=self.dtype)
        self.visual_projection = nn.Dense(bert_config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(bert_config.initializer_range, self.dtype))

        self.visual_position_embeddings = nn.Embed(
            bert_config.max_position_embeddings,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=bert_config.initializer_range),
            dtype=self.dtype,
        )
        self.visual_token_type_embeddings = nn.Embed(
            bert_config.type_vocab_size,
            bert_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=bert_config.initializer_range),
            dtype=self.dtype,
        )

        self.LayerNorm = nn.LayerNorm(epsilon=bert_config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=bert_config.hidden_dropout_prob)
        
    def __call__(self, input_ids, token_type_ids, position_ids, pixel_values, visual_token_type_ids, visual_position_ids, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        word_embeddings = inputs_embeds + token_type_embeddings + position_embeds

        # Visual Embed
        visual_inputs_embeds = self.vit_module(pixel_values=pixel_values)[0]
        visual_inputs_embeds = self.visual_projection(visual_inputs_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids.astype("i4"))
        visual_position_embeds = self.visual_position_embeddings(visual_position_ids.astype("i4"))

        # Sum all visual embeddings
        visual_embeddings = visual_inputs_embeds + visual_token_type_embeddings + visual_position_embeds

        # Concat
        hidden_states = jnp.concatenate((word_embeddings, visual_embeddings),axis=1)

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states
      
class FlaxViTBertModule(nn.Module):
    config: ViTBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxViTBertEmbeddings(self.config, dtype=self.dtype)
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
            input_ids, token_type_ids, position_ids, pixel_values, visual_token_type_ids, visual_position_ids, deterministic=deterministic
        )

        combined_attention_mask = jnp.concatenate((attention_mask, visual_attention_mask), axis=1)

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
      
      
class FlaxViTBertModel(FlaxPreTrainedModel):
    config_class: ViTBertConfig
    module_class = FlaxViTBertModule

    def __init__(
        self, config: ViTBertConfig, input_shape: Tuple = None, seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):

        if input_shape is None:
            input_shape = ((1, 1), (1, config.vit_config.image_size, config.vit_config.image_size, 3), (1, 197))

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        textual_input_shape = input_shape[0]
        input_ids = jnp.zeros(textual_input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), textual_input_shape)
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])
        visual_attention_mask = jnp.ones(input_shape[2]) # TODO: Fix this
        visual_token_type_ids = jnp.ones(input_shape[2]) # TODO: Fix this
        visual_position_ids = jnp.broadcast_to(jnp.zeros(jnp.atleast_2d(input_ids).shape[-1]), input_shape[2]) # TODO: Fix this

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, pixel_values,
        visual_attention_mask,
        visual_token_type_ids, 
        visual_position_ids, return_dict=False)[
            "params"
        ]

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
        output_attentions = output_attentions if output_attentions is not None else self.config.bert_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.bert_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.bert_config.return_dict


        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if visual_token_type_ids is None:
            visual_token_type_ids = jnp.ones(input_ids.shape) # TODO: Fix this.

        if visual_position_ids is None:
            visual_position_ids = jnp.broadcast_to(jnp.atleast_2d(input_ids).shape[-1],input_ids.shape) # TODO: Fix this.

        if visual_attention_mask is None:
            visual_attention_mask = jnp.ones(input_ids.shape) # TODO: Fix this.

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
    def from_bert_vit_pretrained(
        cls,
        bert_model_name_or_path: str = None,
        vit_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:

        kwargs_bert = {
            argument[len("bert_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        kwargs_vit = {
            argument[len("vit_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_bert.keys():
            del kwargs["bert_" + key]
        for key in kwargs_vit.keys():
            del kwargs["vit_" + key]

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

        vit_model = kwargs_vit.pop("model", None)
        if vit_model is None:
            assert (
                vit_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `vit_model_name_or_path` has to be defined"
            from transformers import FlaxViTModel

            if "config" not in kwargs_vit:
                from transformers import ViTConfig

                vit_config = ViTConfig.from_pretrained(vit_model_name_or_path)
                kwargs_vit["config"] = vit_config

            vit_model = FlaxViTModel.from_pretrained(vit_model_name_or_path, *model_args, **kwargs_vit)

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = ViTBertConfig.from_bert_vit_configs(bert_model.config, vit_model.config, **kwargs)

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        for key in model.params.keys():
            if key != "embeddings":
                model.params[key] = bert_model.params[key]
            else:
                model.params["embeddings"]["vit_module"] = vit_model.params
                for sub_key in bert_model.params[key]:
                    model.params[key][sub_key] = bert_model.params[key][sub_key]

        return model
      
# Usage
# >>> flax_model = FlaxViTBertModel.from_bert_vit_pretrained('bert-base-uncased', 'google/vit-base-patch16-224-in21k', seed=0, dtype=jnp.float32)
# >>> outputs = flax_model(input_ids, attention_mask,token_type_ids, position_ids, pixel_values, visual_attention_mask, visual_token_type_ids, visual_position_ids, output_hidden_states=True)
