import numpy as np
from copy import deepcopy
import random
import pickle
import os
import matplotlib.pyplot as plt

from core import TangoBoard


def valid_partial_line(tango_line):
    prev_cell = tango_line[0]
    count = 1
    for cell in tango_line[1:]:
        if prev_cell == cell:
            count += 1
            if count == 3:
                return False
        else:
            count = 1
        prev_cell = cell
    
    num_zeros = (tango_line == 0).sum()
    if num_zeros > 3:
        return False
    
    num_ones = (tango_line == 1).sum()
    if num_ones > 3:
        return False
    
    return True


def valid_partial(tango_board, row, col):
    tango_board.apply_equal_rule(row, col)
    tango_board.apply_opp_rule(row, col)
    # Check consecutive values
    if (row, col + 1) in tango_board.fixed_cells:
        tango_row = tango_board.board[row, :col + 2]
    else:
        tango_row = tango_board.board[row, :col + 1]
    if (row + 1, col) in tango_board.fixed_cells:
        tango_col = tango_board.board[:row + 2, col]
    else:
        tango_col = tango_board.board[:row + 1, col]
    if valid_partial_line(tango_row) and valid_partial_line(tango_col):
        return True
    else:
        return False


def recursive_tango(position, tango_board, solutions, start = 0, check_counts = 0):
    if start == position:
        check_counts += 1
        # if check_counts % 10 == 0:
        # print(f"{check_counts} possibilited checked")
        if tango_board.check():
            # print('Solution added!')
            solutions.append(deepcopy(tango_board))
        return tango_board, solutions, check_counts
    pos = start
    row, column = divmod(pos, 6)
    if not (row, column) in tango_board.fixed_cells:
        for value in [0, 1]:
            tango_board.set_cell(row, column, value)
            if valid_partial(tango_board, row, column):
                tango_board, solutions, check_counts = recursive_tango(position,
                                                                    tango_board,
                                                                    solutions,
                                                                    start=pos + 1,
                                                                    check_counts=check_counts)
    else:
        tango_board, solutions, check_counts = recursive_tango(position,
                                                                tango_board,
                                                                solutions,
                                                                start=pos + 1,
                                                                check_counts=check_counts)

    return tango_board, solutions, check_counts


def compare_sols(board_1, board_2):
    return (board_1 == board_2).all()


def get_unique_sols(sols):
    unique_sols = [sols[-1]]
    for n in range(len(sols) - 1):
        # print(f"Checking solution {n}...")
        # print(f"Comparing with {len(list(range(n + 1, len(sols))))}")
        sol_1 = sols[n]
        for m in range(n + 1, len(sols)):
            sol_2 = sols[m]
            equal_sols = compare_sols(sol_1.board, sol_2.board)
            if equal_sols:
                print(f"Solution {n} is repeated!")
                break
        if not equal_sols:
            unique_sols.append(sol_1)
    return unique_sols


def remove_cells(board: TangoBoard, n_cells: int):
    cells = [divmod(n, 6) for n in range(36)]
    removed_cells = list()
    for n in range(n_cells):
        cell = random.choice(cells)
        # print(f"Removing {cell} with value {board.get_cell(cell[0], cell[1])}")
        board.set_cell(cell[0], cell[1], -1)
        cells.remove(cell)
        removed_cells.append(cell)
    for cell in cells:
        board.fixed_cells.append(cell)
    return board, removed_cells


def reduce_and_reconstruct(tango_board, n_cells):
    # print(f"Removing {n_cells} cells...")
    new_board, _ = remove_cells(deepcopy(tango_board),
                                            n_cells)
    reduced_board = deepcopy(new_board)
    _, sols, _ = recursive_tango(position=36,
                                            tango_board=new_board,
                                            solutions=list())
    n_sols = len(sols)
    if n_sols == 0:
        unique_sols = list()
    else:
        unique_sols = get_unique_sols(sols)
    return unique_sols, reduced_board


def find_minimum_board(tango_board):
    for n_cells in range(1, 37):    
        unique_sols, reduced_board = reduce_and_reconstruct(tango_board, n_cells)
        if len(unique_sols) != 1:
            return n_cells - 1, reduced_board
    return n_cells, reduced_board


def save_solution(board, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(board, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_solution(filepath):
    with open(filepath, 'rb') as handle:
        board = pickle.load(handle)
    return board


def save_sols(sols, folder):
    os.makedirs(folder, exist_ok=True)
    for i, sol in enumerate(sols):
        filepath = os.path.join(folder, 'sol_' + str(i + 1) + '.pickle')
        save_solution(sol, filepath)


def load_sols(folder):
    sols = list()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        sols.append(load_solution(filepath))
    return sols


if __name__ == '__main__':
    # # # Fixed solution
    # fill_board = np.array([[-1, -1, -1,  0, -1, -1],
    #                        [ 1,  1, -1, -1, -1, -1],
    #                        [-1, -1, -1,  0, -1, -1],
    #                        [-1, -1, -1, -1,  1, -1],
    #                        [ 0,  0, -1, -1, -1, -1],
    #                        [-1, -1,  1, -1, -1,  1]])
    # eq_rules = []
    # opp_rules = [[(0, 3), (1, 3)], [(0, 4), (1, 4)], [(3, 3), (4, 3)]]
    # board = TangoBoard()
    # board.fulfill(fill_board, eq_rules, opp_rules)
    # print('Original board:')
    # board.print()
    # tango_board, sols, check_counts = recursive_tango(position=36,
    #                                                   tango_board=board,
    #                                                   solutions=list())
    # unique_sols = get_unique_sols(sols)
    # n_sols = len(sols)
    # n_unique = len(unique_sols)
    # print('Solutions:', n_sols)
    # print('Unique solutions:', n_unique)
    # if n_unique != 0:
    #     unique_sols[0].print()
    # elif n_unique != 1:
    #     unique_sols[-1].print()

    # # Find solutions with random cells and rules
    # random_cells = 5
    # random_rules = 4
    # max_iter = 20
    # sols_array = np.empty(max_iter)
    # for i in range(max_iter):
    #     board = TangoBoard(random_cells=random_cells,
    #                        random_rules=random_rules)
    #     print(f"{random_cells} random fixed cells, {board.fixed_cells[:random_cells]}")
    #     print(f"{len(board.equal_rules)} random equal rules, {board.equal_rules}")
    #     print(f"{len(board.opp_rules)} random opp rules, {board.opp_rules}")
    #     # board.print()
    #     tango_board, sols, check_counts = recursive_tango(position=36,
    #                                                     tango_board=board,
    #                                                     solutions=list())
    #     n_sols = len(sols)
    #     if n_sols == 0:
    #         unique_sols = list()
    #     else:
    #         unique_sols = get_unique_sols(sols)
    #     n_unique = len(unique_sols)
    #     print('Solutions:', n_sols)
    #     print('Unique solutions:', n_unique)
    #     if n_unique == 0:
    #         tango_board.print()
    #     elif n_unique == 1:
    #         unique_sols[0].print()
    #     else:
    #         unique_sols[-1].print()
    #     sols_array[i] = n_sols
    #     print('*****************')
    
    # avg_sols = sols_array.mean()
    # min_sols = sols_array.min()
    # max_sols = sols_array.max()

    # print("Minimum number of solutions:", min_sols)
    # print("Maximum number of solutions:", max_sols)
    # print("Average number of solutions:", avg_sols)

    # # Save solutions
    # board = TangoBoard()
    # tango_board, sols, check_counts = recursive_tango(position=36,
    #                                                 tango_board=board,
    #                                                 solutions=list())
    # n_sols = len(sols)
    # if n_sols == 0:
    #     unique_sols = list()
    # else:
    #     unique_sols = get_unique_sols(sols)
    # n_unique = len(unique_sols)
    # print('Solutions:', n_sols)
    # print('Unique solutions:', n_unique)
    # save_sols(unique_sols, 'solutions')

    # # Load and reconstruct solution
    n_unique = len(os.listdir('solutions'))  
    n_cells = 10
    array_n_cells = np.empty(n_unique)
    for i in range(n_unique):
        filepath = os.path.join('solutions',
                                'sol_' + str(i + 1) + '.pickle')
        board = load_solution(filepath)
        n_cells, reduced_board = find_minimum_board(board)
        initial_cells = 36 - n_cells
        array_n_cells[i] = initial_cells
        # print(f"Board {i} can be reconstructed with {initial_cells} cells.")
        # reduced_board.print()
        # print()
    
    print()
    print('Minimum of n_cells:', array_n_cells.min())
    print('Maximum of n_cells:', array_n_cells.max())
    print('Average of n_cells:', array_n_cells.mean())
    print('Median of n_cells:', np.median(array_n_cells))

    plt.hist(array_n_cells, bins=10)
    plt.savefig('hist_initial_cells.png')
