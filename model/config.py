from detectron2.config import CfgNode as CN

def add_config(cfg):
    """
    yaml内的参数必须先定义才可以使用 cfg.merge_from_file()
    """
    
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.DATASET_MAPPER_NAME =  "coco_panoptic_lsj"