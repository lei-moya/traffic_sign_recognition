import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 定义框选数据集类
class Dataset(Dataset):
    def __init__(self, image_paths, label_df, target_size, num=5):
        """
        Args:
            image_paths (list): 图片路径列表。
            label_df (pd.Dataframe): 标签集字典。
            target_size (tuple): 缩放目标尺寸。
            num (int): 所需列数。
        """
        self.image_paths = image_paths
        self.label_df = label_df
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        self.num = num

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        def process_variable(var):
            if isinstance(var, torch.Tensor):
                # 如果是张量，使用 .item() 获取标量值
                return var.item()
            else:
                # 如果是整数，直接使用
                return var

        image_path = self.image_paths[idx]

        box_label = self.label_df.iloc[process_variable(idx), 0:4].values.astype('float').reshape(-1, 4)
        image = Image.open(image_path).convert("RGB")  # 确保图像为三通道

        # 计算缩放比例
        scale_x = self.target_size[0] / image.width
        scale_y = self.target_size[1] / image.height

        # 对图像进行缩放
        image = self.transform(image)

        # 对标签的边界框坐标进行缩放
        box_label[:, [0, 2]] *= scale_x  # 缩放x坐标
        box_label[:, [1, 3]] *= scale_y  # 缩放y坐标

        if self.num == 5:
            class_label = self.label_df.iloc[process_variable(idx), 4].astype('float')
            return image, box_label.squeeze(), class_label
        return image, box_label.squeeze()