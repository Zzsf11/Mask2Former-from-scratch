import torch
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data import dataset_mapper
from detectron2.data import MetadataCatalog, build_detection_train_loader

from dataset.datamapper import COCOPanopticNewBaselineDatasetMapper
from config import add_config


class Trainer(DefaultTrainer):
    @classmethod # 这个方法在类没有被实例化的情况就可以直接通过 `Trainer.build_train_loader`调用
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_panoptic_lsj':
            mapper =  COCOPanopticNewBaselineDatasetMapper
            return build_detection_train_loader(cfg, mapper = mapper)
    
    
def setup(args):
    """
    解析args
    """
    cfg = get_cfg() # 得到一个预设的config值
    add_config(cfg) # 将新的key先加入到cfg中
    cfg.merge_from_file(args.config_file) # 使用指定的yaml文件
    cfg.merge_from_list(args.opts) # 使用args内的参数
    cfg.freeze() # 防止被意外更改
    default_setup(cfg, args) # 预设的logger设置以及相关envs
    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg
    
def main(args):
    cfg = setup(args)
    
    train = Trainer(cfg)
    dataloader = train.build_train_loader(cfg)
    return  

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )