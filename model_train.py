import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import target_size
from data_preprocess import Dataset
from labelDF import collect_label_files, collect_ends_files
from model import CNN


def calculate_iou(pred_boxes, true_boxes):
    # 计算交集的坐标
    inter_x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

    # 计算交集的面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算预测边界框和真实边界框的面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    # 计算并集的面积
    union_area = pred_area + true_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou


def adjust_coordinates(predicts, images):
    batch_size, channels, height, width = images.shape
    predicts[:, [0, 2]] *= width / target_size[0]  # 调整 x 坐标
    predicts[:, [1, 3]] *= height / target_size[1]  # 调整 y 坐标
    return predicts


def model_train():
    # 数据加载和预处理
    train_image_paths = collect_ends_files(directory="train")
    train_label_df = collect_label_files(directory="train", cols=['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])

    dataset = Dataset(image_paths=train_image_paths, label_df=train_label_df, target_size=target_size)

    # 定义训练集和验证集的大小
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 随机分割数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DataLoader实例
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型和优化器
    model = CNN()
    class_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, box_labels, class_labels) in enumerate(train_loader):

            images = images.float()
            box_labels = box_labels.float()
            class_labels = class_labels.long()

            # 前向传播
            class_outputs, box_outputs = model(images)
            loss = class_criterion(class_outputs, class_labels) + box_criterion(adjust_coordinates(box_outputs, images),
                                                                                box_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    return model, val_dataset, num_epochs


def model_eval(model, val_dataset, num_epochs):
    val_image_paths = collect_ends_files(directory="test")
    val_label_df = collect_label_files(directory="test", cols=['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'])

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # 评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        iou_total = 0
        for images, box_labels, class_labels in val_loader:
            class_outputs, box_outputs = model(images)
            _, predicted = torch.max(class_outputs.data, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
            iou = calculate_iou(adjust_coordinates(box_outputs, images), box_labels)
            iou_total += iou.mean().item()

    print(f'Accuracy of the model on the validation images: {100 * correct / total}%')
    print(f'Average IoU of the model on the validation images: {100 * iou_total / len(val_loader)}')

    with torch.no_grad():
        iou_total = 0

        test_dataset = Dataset(image_paths=val_image_paths, label_df=val_label_df, target_size=target_size, num=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for images, box_labels in test_loader:
            _, box_outputs = model(images)
            iou = calculate_iou(adjust_coordinates(box_outputs, images), box_labels)
            iou_total += iou.mean().item()

    print(f'Average IoU of the model on the test images: {100 * iou_total / len(test_loader)}')

    torch.save(model, f'model_{num_epochs}_{100 * correct / total}_{100 * iou_total / len(test_loader)}.pth')

# model, val_dataset, num_epochs = model_train()
# model_eval(model, val_dataset, num_epochs)
