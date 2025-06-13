import torch
from PIL import Image
from matplotlib import pyplot as plt, patches
from torchvision import transforms



def adjust_coordinates(predicts, images):
    channels, height, width = images.shape
    predicts[:, [0, 2]] *= width / 32  # 调整 x 坐标
    predicts[:, [1, 3]] *= height / 32  # 调整 y 坐标
    return predicts

for i in ['2025-06-12_111645.png','2025-06-12_111703.png','2025-06-12_111716.png','2025-06-12_113359.png','2025-06-12_113412.png']:
    image = Image.open(i).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    model = torch.load('model_15_98.40465465465465_91.99480093480493.pth', weights_only=False)
    model.eval()
    class_outputs, box_outputs = model(image)
    _, class_predicted = torch.max(class_outputs.data, 1)
    box_predicted = adjust_coordinates(box_outputs, image)
    image = image.permute(1, 2, 0).numpy()  # 将图像从 (C, H, W) 转换为 (H, W, C)
    box = box_predicted[0].detach().numpy()  # 获取预测的边界框
    class_pred = class_predicted.item()  # 获取预测的类别

    # 创建绘图对象
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 绘制边界框
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=5, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)

    # 显示预测的类别
    ax.set_title(f'Class: {class_pred}', color='r', size=40)

    # 显示图像
    plt.show()
