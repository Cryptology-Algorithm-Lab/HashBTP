from easydict import EasyDict as edict

cfg = edict()

# Image Directory for Benchmark Datasets
cfg.img_dir = "Your Image Directory"

# Configs for pre-trained FR models
# Format: (arch, param_dir, is_onnx, name)
#     -arch:"r50", "r100", "vit", "vitkprpe"
#     -param_dir:  /* Your Parameter directories */; the followings are default.
#     -is_onnx: if true, then we directly load the FR model from onnx.
#     -name: {Source}_{Arch}_{TrainDataset}_{SuffixLabel}

# From InsightFace
cfg.model1 = ("r50", "./params/InsightFace/r50_glint_arcface.onnx", True, "InsightFace_r50_glint_ArcFace")
cfg.model2 = ("r50", "./params/InsightFace/r50_webface_arcface.onnx", True, "InsightFace_r50_WebFace600K_ArcFace")
cfg.model3 = ("r100", "./params/InsightFace/r100_ms1mv3_arcface.onnx", True, "InsightFace_r100_ms1mv3_ArcFace")

# From OpenSphere
cfg.model4 = ("sf20", "./params/OpenSphere/sf20_vggface2_spface2.pth", False, "OpenSphere_sf20_vggface2_SphereFace2")
cfg.model5 = ("sf64", "./params/OpenSphere/sf64_ms1mv2_spplus.pth", False, "OpenSphere_sf64_ms1m_SphereFacePlus")
cfg.model6 = ("r100", "./params/OpenSphere/r100_ms1mv2_spherer.pth", False, "OpenSphere_r100_ms1mv2_SphereFaceR")

# From CVLFace
cfg.model7 = ("r100", "./params/CVLFace/ir101_webface12m_adaface.pt", False, "CVLFace_ir101_webface12m_AdaFace")
cfg.model8 = ("vit", "./params/CVLFace/vit_webface4m_adaface.pt", False, "CVLFace_vit_webface4m_AdaFace")
cfg.model9 = ("vitkprpe", "./params/CVLFace/vitkprpe_webface4m_adaface.pt", False, "CVLFace_vitkprpe_webface4m_AdaFace")

# From UniTSFace
cfg.model10 = ("r50", "./params/Misc/r50_webface_unitsface.onnx", True, "UniTSFace_r50_webface_UniTSFace")

# From ElasticFace
cfg.model11= ("r100", "./params/Misc/r100_ms1mv2_elasticface.pth", False, "ElasticFace_r100+ms1mv2_ElasticFaceArc+")

# From Face-Transformer
cfg.model12 = ("vit", "./params/Misc/vit_ms1mv3_cosface.pth", False, "Face-Transformer_vit_ms1mv3_CosFace")
