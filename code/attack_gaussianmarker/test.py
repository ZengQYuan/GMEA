import numpy as np
import geatpy as ea


class PrizeOptimization(ea.Problem):
    def __init__(self, DIM=5000):
        name = 'Maximize_Winners'
        M = 1  # 单目标
        maxormins = [-1]  # 最大化目标（-1表示最大化）
        varTypes = [0] * DIM  # 连续型变量
        lb = [0] * DIM  # 变量下界
        ub = [1] * DIM  # 变量上界
        lbin = [1] * DIM  # 包含下边界
        ubin = [1] * DIM  # 包含上边界
        ea.Problem.__init__(self, name, M, maxormins, DIM, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 获取种群决策变量矩阵
        Phen = pop.Phen  # 形状为(NIND, DIM)

        # 计算每个个体中小于0.5的元素数量（中奖人数）
        winners = np.sum(Phen < 0.5, axis=1)

        # 将目标函数值赋值给种群对象
        pop.ObjV = winners.reshape(-1, 1)  # 转换为列向量


if __name__ == "__main__":
    # 实例化问题对象（10000维）
    problem = PrizeOptimization(DIM=20000)

    # 算法配置
    algorithm = ea.soea_DE_rand_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),  # 实数编码，种群规模50
        MAXGEN=5000,  # 最大进化代数
        # trappedValue=1e-6,  # 收敛阈值
    )
    algorithm.mutOper.F = 0.15  # 设置变异因子（默认0.5，范围0-2）[3,6](@ref)
    algorithm.recOper.XOVR = 0.15  # 设置交叉概率（默认0.5，范围0-1）[3,6](@ref)

    # 优化求解
    res = ea.optimize(
        algorithm,
        verbose=True,
        drawing=1,
        outputMsg=False,
        drawLog=True,
        saveFlag=False
    )

    # 输出最优解
    best_x = res['Vars'][0]
    print(f"最大中奖人数: {int(res['ObjV'][0][0])}")
    print(f"前10个元素示例: {best_x[:10].round(3)}")