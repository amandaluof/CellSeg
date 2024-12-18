from torchvision.models import resnet50
from thop import profile
# 增加可读性
from thop import clever_format
from models.hovernet.net_desc import HoVerNet
import torch

# 可替换为自己的模型及输入
model = HoVerNet(input_ch=3, nr_types=6, freeze=False, mode='fast')
inputs = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(inputs, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

print(flops, params)