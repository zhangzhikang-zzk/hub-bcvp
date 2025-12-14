import numpy as np


def jaccard_similarity(string1, string2):
    """
    基于Jaccard系数计算两个字符串的相似度
    
    Jaccard相似度 = |A ∩ B| / |A ∪ B|
    其中A和B分别是两个字符串中字符的集合
    
    参数:
        string1: 第一个字符串
        string2: 第二个字符串
    
    返回:
        Jaccard相似度（浮点数，范围在[0,1]之间）
    """
    # 将字符串转换为字符集合
    set1 = set(string1)
    print(f"set1: {set1}")
    set2 = set(string2)
    print(f"set2: {set2}")
    
    # 计算交集和并集
    intersection = set1.intersection(set2)
    print(f"intersection: {intersection}")
    union = set1.union(set2)
    print(f"union: {union}")
    
    # 避免除零错误
    if len(union) == 0:
        return 1.0  # 两个空字符串应该完全相似
    
    # 计算Jaccard相似度
    return len(intersection) / len(union)


# 示例用法
if __name__ == "__main__":
    s1 = "kittengasd"
    s2 = "sittingasdfg"
   
    print(jaccard_similarity(s1, s2))