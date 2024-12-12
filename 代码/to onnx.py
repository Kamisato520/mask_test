import torch
from load_model.pytorch_loader import load_pytorch_model

# 加载模型
model = load_pytorch_model('./models/model360.pth')
model.eval()  # 将模型设置为评估模式

# 创建一个示例输入张量。输入尺寸应与您训练模型时使用的尺寸相匹配。
# 例如，如果您的模型输入是 3 通道的 160x160 图像，则输入张量的形状应为 (1, 3, 160, 160)
dummy_input = torch.randn(1, 3, 224, 224)  # 这里的尺寸需要根据您的模型进行调整

# 导出模型
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'])

print("Model has been converted to ONNX format and saved as model.onnx")
