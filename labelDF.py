from imagePath import collect_ends_files
import pandas as pd

"""
    Args:
        directory (string): 标签集目录。
        cols (list): 需要使用的标签列。
    函数用例：
        如需获取 train 标签集 =>
            directory = 'test'
            cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']
            train_labels = collect_label_files(directory, cols)
"""

def collect_label_files(directory, cols):
    Label_paths = collect_ends_files(directory, ends=".csv")
    dataframes = []
    for path in Label_paths:
        label_n = pd.read_csv(path, delimiter=';', usecols=cols)
        dataframes.append(label_n)
    label_df = pd.concat(dataframes, ignore_index=True)
    return label_df
