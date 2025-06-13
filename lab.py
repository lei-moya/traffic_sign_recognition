import torch
from matplotlib import pyplot as plt, patches
from torch.utils.data import Subset, DataLoader
from config import target_size
from data_preprocess import Dataset
from labelDF import collect_label_files, collect_ends_files
from model_train import adjust_coordinates

train_image_paths = collect_ends_files(directory="train")
train_label_df = collect_label_files(directory="train", cols=['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])

dataset = Dataset(image_paths=train_image_paths, label_df=train_label_df, target_size=target_size)

random_indices = torch.randperm(len(dataset))[:10]

# 创建子数据集
subset = Subset(dataset, random_indices)

# 可以使用 DataLoader 来迭代子数据集
data_loader = DataLoader(subset, batch_size=10, shuffle=False)

# 加载模型权重
model = torch.load('model_15_98.40465465465465_91.99480093480493.pth', weights_only=False)

# 设置模型为评估模式
model.eval()

# 获取样本
for images, box_labels, class_labels in data_loader:
    class_outputs, box_outputs = model(images)
    _, class_predicted = torch.max(class_outputs.data, 1)
    box_predicted = adjust_coordinates(box_outputs, images)

    # 绘制每个图像及其预测的边界框
    for i in range(images.size(0)):
        image = images[i].permute(1, 2, 0).numpy()  # 将图像从 (C, H, W) 转换为 (H, W, C)
        box = box_predicted[i].detach().numpy()  # 获取预测的边界框
        class_pred = class_predicted[i].item()  # 获取预测的类别

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