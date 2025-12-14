import numpy as np


def edit_distance(string1, string2):
    """
    使用动态规划计算两个字符串之间的编辑距离。
    
    参数:
        string1: 第一个字符串
        string2: 第二个字符串
    
    返回:
        编辑距离（整数）
    """
    m, n = len(string1), len(string2)
    print(f"m: {m}, n: {n}")
    # m 和 n 加 1 是为了处理空字符串的情况
    # 创建 (m+1) x (n+1) 的矩阵，初始化为 0
    matrix = np.zeros((m + 1, n + 1), dtype=int)
    print(f"Initial matrix:\n{matrix}")

    # 初始化第一行和第一列
    for i in range(m + 1):
        matrix[i][0] = i
        # print(f"matrix[{i}][0]\n: {matrix}")
    for j in range(n + 1):
        matrix[0][j] = j
        # print(f"matrix[0][{j}]\n: {matrix}")
    print(f"Initial2 matrix:\n{matrix}") 

    # 填充矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i - 1] == string2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
                # print(f"Initial[if] matrix:\n{matrix}") 
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,      # 删除
                    matrix[i][j - 1] + 1,      # 插入
                    matrix[i - 1][j - 1] + 1   # 替换
                )
                # print(f"Initial[else] matrix:\n{matrix}") 

    return matrix[m][n]


def similarity_based_on_edit_distance(string1, string2):
    """
    基于编辑距离计算两个字符串的相似度（归一化到 [0,1] 区间）
    """
    ed = edit_distance(string1, string2)
    print(f"编辑距离: {ed}")  # 输出: 3
    max_len = max(len(string1), len(string2))
    return 1 - ed / max_len


# 示例用法
if __name__ == "__main__":
    s1 = "kittengasd"
    s2 = "sittingasdfg"
   
    print(similarity_based_on_edit_distance(s1, s2))  # 输出0.5714285714285714