from .iresnet import *
from .sfnet import *
from .mobilefacenet import *
from .vit import *

class Rescale(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        assert(high > low)
        self.low = low
        self.high = high
    
    def forward(self, x):
        x = (x + 1) / 2
        return x * (self.high - self.low) + self.low

class dummy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.module = net
    def forward(self, x):
        return self.module(x) 
    
class dummy_cvl(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model
        
    def forward(self, x):
        return self.net(x)

def get_backbone(config, device, **kwargs):
    arch, param_dir, is_onnx, name = config
    
    if "/CVLFace/" in param_dir:
        return get_backbone_cvl(config, device)
    
    if not is_onnx:
        if arch == "r50":
            net = iresnet50(**kwargs)
            
        elif "OpenSphere" in param_dir and arch == "r100":
            net = iresnet100(**kwargs)
            net = dummy(net)
        elif arch == "r100":
            net = iresnet100(**kwargs)            
        elif arch == "sf20":
            net = sfnet20(**kwargs)
            net = dummy(net)
        elif arch == "sf64":
            net = sfnet64(**kwargs)
            net = dummy(net)
        elif arch == "vit":
            net = get_vit()
        else:
            raise ValueError(f"Architecture {arch} does not supported.")

        net.load_state_dict(torch.load(param_dir, map_location ="cpu"))

        if arch == "vit":
            net = nn.Sequential(Rescale(0, 255), net)
            
        if "magface" in param_dir:
            net = nn.Sequential(Rescale(0, 1), net)

        net = net.to(device).eval()
        return net
    
    else:
        from onnx2torch import convert
        net = convert(param_dir)
        net = net.to(device).eval()
        return net
    
def get_backbone_cvl(config, device, **kwargs):
    arch, param_dir, is_onnx, name = config
    if "ir101" in param_dir:
        from .CVLFace.iresnet.model import IR_101
        model = IR_101([112,112])
        model = dummy_cvl(model)
        params = torch.load(param_dir, map_location = "cpu")
        model.load_state_dict(params)
        model = model.eval().to(device)
        return model
    
    elif arch=="vit":
        from .CVLFace.vit.vit import VisionTransformer
        model = VisionTransformer(
            img_size = 112, patch_size = 8, num_classes = 512, embed_dim = 512,
            depth = 24, mlp_ratio = 3, num_heads = 16, drop_path_rate = 0.1, norm_layer = "ln", 
            mask_ratio = 0.0
        )
        model = dummy_cvl(model)
        params = torch.load(param_dir, map_location = "cpu")
        model.load_state_dict(params)
        model = model.eval().to(device)
        return model
        
    