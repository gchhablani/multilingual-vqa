import copy

from transformers import BertConfig, ViTConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ViTBertConfig(PretrainedConfig):
    model_type = "vit-bert"
    is_composition = True

    def __init__(self, bert_config_dict, vit_config_dict, **kwargs):
        super().__init__(**kwargs)

        if bert_config_dict is None:
            raise ValueError("`bert_config_dict` can not be `None`.")

        if vit_config_dict is None:
            raise ValueError("`vit_config_dict` can not be `None`.")

        self.bert_config = BertConfig(**bert_config_dict)

        self.vit_config = ViTConfig(**vit_config_dict)

    @classmethod
    def from_bert_vit_configs(
        cls, bert_config: PretrainedConfig, vit_config: PretrainedConfig, **kwargs
    ):
        return cls(
            bert_config_dict=bert_config.to_dict(),
            vit_config_dict=vit_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["bert_config"] = self.bert_config.to_dict()
        output["vit_config"] = self.vit_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
