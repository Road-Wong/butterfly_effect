import numpy as np

def initialize_custom_board(rows, cols, initial_directions):
    """
    初始化自定义游戏板。
    
    参数：
    rows, cols: 游戏板的行数和列数
    initial_directions: 二维列表，指定每个表盘的初始方向。-1 表示缺失的表盘。
    
    返回：
    初始化后的游戏板（numpy 数组）
    """
    board = np.array(initial_directions, dtype=int)
    assert board.shape == (rows, cols), "初始方向的尺寸必须与游戏板一致"
    return board

def get_next_position(x, y, direction, rows, cols):
    """
    根据当前表盘的方向，计算相邻表盘的位置。
    """
    if direction == 0:  # 向上
        return x - 1, y
    elif direction == 1:  # 向右
        return x, y + 1
    elif direction == 2:  # 向下
        return x + 1, y
    elif direction == 3:  # 向左
        return x, y - 1

def rotate_dial(board, x, y, rows, cols):
    """
    旋转表盘，并触发连锁反应。
    """
    global total_rotations
    
    if board[x, y] == -1:  # 跳过缺失的表盘
        return
    
    board[x, y] = (board[x, y] + 1) % 4
    total_rotations += 1
    
    direction = board[x, y]
    next_x, next_y = get_next_position(x, y, direction, rows, cols)
    
    if 0 <= next_x < rows and 0 <= next_y < cols and board[next_x, next_y] != -1:
        rotate_dial(board, next_x, next_y, rows, cols)

def play_game(board, moves):
    """
    执行游戏操作。
    
    参数：
    board: 初始化的游戏板
    moves: 拨盘动作列表，格式为[(x1, y1), (x2, y2), ...]，坐标从0开始
    
    返回：
    总旋转次数
    """
    global total_rotations
    total_rotations = 0
    
    rows, cols = board.shape
    
    for move in moves:
        x, y = move
        if 0 <= x < rows and 0 <= y < cols and board[x, y] != -1:
            rotate_dial(board, x, y, rows, cols)
    
    return total_rotations

# 使用示例
if __name__ == "__main__":
    # 创建一个 5x4 的游戏板，带有自定义初始方向和缺失表盘
    initial_directions = [
        [0, -1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, -1, 0]
    ]
    
    custom_board = initialize_custom_board(5, 4, initial_directions)
    print("初始游戏板：")
    print(custom_board)
    
    moves = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    total_rotations = play_game(custom_board, moves)
    
    print("\n执行操作后的游戏板：")
    print(custom_board)
    print(f"\n总旋转次数：{total_rotations}")