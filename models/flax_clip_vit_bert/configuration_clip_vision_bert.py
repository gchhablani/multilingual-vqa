import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import BertConfig, CLIPVisionConfig


logger = logging.get_logger(__name__)


class CLIPVisionBertConfig(PretrainedConfig):
    model_type = "clip-vision-bert"
    is_composition = True

    def __init__(self, bert_config_dict, clip_vision_config_dict, **kwargs):
        super().__init__(**kwargs)

        if bert_config_dict is None:
            raise ValueError("`bert_config_dict` can not be `None`.")

        if clip_vision_config_dict is None:
            raise ValueError("`clip_vision_config_dict` can not be `None`.")

        self.bert_config = BertConfig(**bert_config_dict)

        self.clip_vision_config = CLIPVisionConfig(**clip_vision_config_dict)

    @classmethod
    def from_bert_clip_vision_configs(cls, bert_config: PretrainedConfig, clip_vision_config: PretrainedConfig, **kwargs):
        return cls(bert_config_dict=bert_config.to_dict(), clip_vision_config_dict=clip_vision_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["bert_config"] = self.bert_config.to_dict()
        output["clip_vision_config"] = self.clip_vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
