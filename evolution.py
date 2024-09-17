import random
import numpy as np
from deap import base, creator, tools, algorithms
import playtable

# 定义自定义游戏板
initial_board = np.array([
        [0, -1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, -1, 0]
    ])

rows, cols = initial_board.shape

# 定义适应度函数
def evaluate(individual):
    # 将个体转换为二维列表
    parameters = []
    for i in range(0, len(individual), 2):
        x, y = individual[i], individual[i+1]
        if 0 <= x < rows and 0 <= y < cols and initial_board[x, y] != -1:
            parameters.append([x, y])
    
    # 调用游戏函数并返回得分
    score = playtable.play_game(initial_board.copy(), parameters)
    return score,

# 创建适应度最大化问题
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 初始化工具箱
toolbox = base.Toolbox()

# 注册随机生成参数的函数
toolbox.register("attr_row", random.randint, 0, rows-1)
toolbox.register("attr_col", random.randint, 0, cols-1)

# 注册个体和种群的创建函数
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_row, toolbox.attr_col), n=10)  # 10 moves per individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册评估函数
toolbox.register("evaluate", evaluate)

# 注册选择、交叉和变异操作
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(rows-1, cols-1), indpb=0.2)

# 定义遗传算法参数
population_size = 100
generations = 200

# 创建初始种群
population = toolbox.population(n=population_size)

# 运行遗传算法
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=True)

# 找到最优个体
best_individual = tools.selBest(population, k=1)[0]
best_parameters = []
for i in range(0, len(best_individual), 2):
    x, y = best_individual[i], best_individual[i+1]
    if 0 <= x < rows and 0 <= y < cols and initial_board[x, y] != -1:
        best_parameters.append([x, y])

# 输出最优参数和得分
best_score = playtable.play_game(initial_board.copy(), best_parameters)
print(f"Best Parameters: {best_parameters}")
print(f"Best Score: {best_score}")