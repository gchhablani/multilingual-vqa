import copy

from transformers import BertConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CLIPVisionBertConfig(PretrainedConfig):

    model_type = "clip-vision-bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "bert_config" not in kwargs:
            raise ValueError("`bert_config` can not be `None`.")

        if "clip_vision_config" not in kwargs:
            raise ValueError("`clip_vision_config` can not be `None`.")

        bert_config = kwargs.pop("bert_config")
        clip_vision_config = kwargs.pop("clip_vision_config")

        self.bert_config = BertConfig(**bert_config)

        self.clip_vision_config = CLIPVisionConfig(**clip_vision_config)

    @classmethod
    def from_clip_vision_bert_configs(
        cls,
        clip_vision_config: PretrainedConfig,
        bert_config: PretrainedConfig,
        **kwargs
    ):
        return cls(
            clip_vision_config=clip_vision_config.to_dict(),
            bert_config=bert_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["clip_vision_config"] = self.clip_vision_config.to_dict()
        output["bert_config"] = self.bert_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
