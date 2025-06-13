import os

"""
    Args:
        directory (string): 图片集目录。
    函数用例：
        如需获取 train 图片集 =>
            directory = 'train'
            trainImage_paths = collect_ends_files(directory)
"""

def collect_ends_files(directory, ends=".ppm"):
    ends_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ends):
                ends_files.append(os.path.join(root, file))
    return ends_files