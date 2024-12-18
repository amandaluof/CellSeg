import argparse

from mmcv import Config

from mmdet.models import build_detector
from mmdet.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[1280, 800],
                        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model,
                           train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    # import torch 
    # import profile
    # from thop import clever_format
    # inputs = torch.randn(1, 3, 1024, 1024)
    # import pdb;pdb.set_trace()
    # flops, params = profile(model, inputs=(inputs, ), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))


if __name__ == '__main__':
    main()
