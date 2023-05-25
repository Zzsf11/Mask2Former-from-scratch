import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone

@META_ARCH_REGISTRY.register() # 将函数注册为 META_ARCHITECTURE
class mask2former(nn.Module):
    @configurable
    def __init__(
      self,
      *,
      backbone: Backbone,
        
    ):
        super().__init__()
        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        
        ret = {
            "backbone": backbone
        }
        
        return ret
    
    def forward(self, batched_inputs):
        
        feature = self.backbone()
        
        return