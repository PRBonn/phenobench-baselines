from re import I


def get_model(cfg):
    model_name = cfg['experiment']['id']
    if model_name == 'maskrcnn':
        from models.maskrcnn import MaskRCNN
        model = MaskRCNN(cfg)
    elif model_name == 'fasterrcnn':
        from models.fasterrcnn import FasterRCNN
        model = FasterRCNN(cfg)
    elif model_name == 'maskrcnn_leaves':
        from models.maskrcnn import MaskRCNN
        model = MaskRCNN(cfg)
    elif model_name == 'fasterrcnn_leaves':
        from models.fasterrcnn import FasterRCNN
        model = FasterRCNN(cfg)
    else:
        raise AttributeError("Model {} not implemented".format(model_name))
    
    return model