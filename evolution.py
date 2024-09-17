import random
import numpy as np
from deap import base, creator, tools, algorithms
import playtable

# 定义自定义游戏板
initial_board = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

rows, cols = initial_board.shape

# 用户可控制的参数
moves_per_individual = 15  # 每个个体的移动次数，用户可以修改这个值

# 定义适应度函数
def evaluate(individual):
    parameters = []
    for i in range(0, len(individual), 2):
        x, y = individual[i], individual[i+1]
        if 0 <= x < rows and 0 <= y < cols and initial_board[x, y] != -1:
            parameters.append([x, y])
    
    score = playtable.play_game(initial_board.copy(), parameters)
    return score,

# 创建适应度最大化问题
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 注册个体和种群的创建函数
toolbox.register("attr_coord", random.randint, 0, max(rows-1, cols-1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_coord, n=moves_per_individual*2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 自定义交叉函数
def cxTwoPointCoordinates(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        
    return ind1, ind2

# 自定义变异函数
def mutUniformCoordinates(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, max(rows-1, cols-1))
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", cxTwoPointCoordinates)
toolbox.register("mutate", mutUniformCoordinates, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义遗传算法参数
population_size = 200
generations = 1000
crossover_prob = 0.7
mutation_prob = 0.2

# 精英保留策略
elite_size = 5

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, verbose=__debug__):
    best_individual = None
    best_score = float('-inf')

    for gen in range(ngen):
        # 评估整个种群
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 更新最佳个体
        current_best = tools.selBest(population, 1)[0]
        if current_best.fitness.values[0] > best_score:
            best_individual = current_best
            best_score = current_best.fitness.values[0]

        # 选择精英
        elite = tools.selBest(population, elite_size)
        
        # 选择下一代个体
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 精英替换
        population = elite + offspring

        # 每10代输出一次统计信息
        if stats is not None and gen % 50 == 0:
            record = stats.compile(population)
            if verbose:
                print(f"Gen {gen}: Avg={record['avg']:.2f}, Min={record['min']:.2f}, Max={record['max']:.2f}")

    return population, best_individual

# 记录统计信息
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# 创建初始种群
population = toolbox.population(n=population_size)

# 运行遗传算法
print("Starting genetic algorithm...")
final_population, best_individual = eaSimpleWithElitism(population, toolbox, cxpb=crossover_prob, 
                                                        mutpb=mutation_prob, ngen=generations, 
                                                        stats=stats, verbose=True)
print("Genetic algorithm completed.")

# 提取最佳移动
best_moves = []
for i in range(0, len(best_individual), 2):
    x, y = best_individual[i], best_individual[i+1]
    if 0 <= x < rows and 0 <= y < cols and initial_board[x, y] != -1:
        best_moves.append([x, y])
best_score = best_individual.fitness.values[0]

# 输出最优参数和得分
print(f"\nBest Score: {best_score}")
print(f"Best Moves: {best_moves}")

# 验证最佳移动
verification_score = playtable.play_game(initial_board.copy(), best_moves)
print(f"Verification Score: {verification_score}")

if verification_score != best_score:
    print("Warning: Verification score does not match best score. This might indicate an issue with the optimization process.")

print(f"\nNumber of moves per individual: {moves_per_individual}")
print(f"Actual number of valid moves found: {len(best_moves)}")
