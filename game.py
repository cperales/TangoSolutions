import numpy as np
from copy import deepcopy
import random


class TangoBoard:
    def __init__(self, n_random=0):
        self.board = -1 * np.ones((6, 6), dtype=np.int8)
        self.fixed_cells = list()
        for _ in range(n_random):
            self.set_cell_random()

    def set_cell_random(self):
        check = False
        while not check:
            possible_cells = [divmod(n, 6) for n in range(36)]
            for cell in self.fixed_cells:
                possible_cells.remove(cell)
            x, y = random.choice(possible_cells)
            value = random.choice([0, 1])
            self.board[x, y] = value
            check = True

            num_zeros_row = (self.board[x, :] == 0).sum()
            if num_zeros_row > 2:
                check = False

            num_ones_row = (self.board[x, :] == 1).sum()
            if num_ones_row > 2:
                check = False

            num_zeros_col = (self.board[:, y] == 0).sum()
            if num_zeros_col > 2:
                check = False

            num_ones_col = (self.board[:, y] == 1).sum()
            if num_ones_col > 2:
                check = False

            if not check:
                self.board[x, y] = -1

        self.fixed_cells.append((x, y))
    
    def get_cell(self, x, y):
        return self.board[x, y]

    def set_cell(self, x, y, value):
        if not (x, y) in self.fixed_cells:
            self.board[x, y] = value
        else:
            print(f"Trying to ")
    
    def change_cell(self, x, y):
        value = self.get_cell(x, y)
        if value == 0:
            self.set_cell(x, y, 1)
        else:
            self.set_cell(x, y, 0)

    def print(self):
        print('---------------\n')
        board = ""
        for i in range(6):
            board += "| "
            for j in range(6):
                board += f"{self.board[i, j]} "
            board += " |\n"
        print(board)
        print('---------------')

    def check_num_col(self, j):
        condition_S = self.board[:, j] == 1
        num_S = np.where(condition_S, 1, 0).sum() == 3
        condition_M = self.board[:, j] == 0
        num_M = np.where(condition_M, 1, 0).sum() == 3
        return num_S * num_M

    def check_num_row(self, i):
        condition_S = self.board[i, :] == 1
        num_S = np.where(condition_S, 1, 0).sum() == 3
        condition_M = self.board[i, :] == 0
        num_M = np.where(condition_M, 1, 0).sum() == 3
        return num_S * num_M
    
    def check_cons_row(self, i):
        condition = True
        prev_cell = self.board[i, 0]
        num = 0
        for cell in self.board[i, 1:]:
            if prev_cell == cell:
                num += 1
            if num == 3:
                condition = False
                break
            prev_cell = cell
        return condition
    
    def check_cons_col(self, j):
        condition = True
        prev_cell = self.board[0, j]
        num = 0
        for cell in self.board[1:, j]:
            if prev_cell == cell:
                num += 1
            if num == 3:
                condition = False
                break
            prev_cell = cell
        return condition

    def fulfill(self):
        if self.board[0, :].sum() == 3 and self.board.min() == 0:
            return True
        return False
    
    def check(self):
        for n in range(6):
            if not self.check_num_row(n) or not self.check_cons_row(n):
                return False
            if not self.check_num_col(n) or not self.check_cons_col(n):
                return False
        return True
    

def valid_partial_line(tango_line):
    count = 0
    for i in range(len(tango_line) - 1):
        if tango_line[i] == tango_line[i + 1]:
            count += 1
            if count == 3:
                return False
    
    num_zeros = (tango_line == 0).sum()
    if num_zeros > 3:
        return False
    
    num_ones = (tango_line == 1).sum()
    if num_ones > 3:
        return False
    
    return True


def valid_partial(tango_board, row, col):
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
        if tango_board.fulfill():
            check_counts += 1
            # if check_counts % 10 == 0:
            # print(f"{check_counts} possibilited checked")
            # tango_board.print()
            if tango_board.check():
                # print('Solution added!')
                # tango_board.print()
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


if __name__ == '__main__':
    import gc
    n = 9
    max_iter = 20
    sols_array = np.empty(max_iter)
    for i in range(max_iter):
        board = TangoBoard(n_random=n)
        print(f"{n} random fixed cells, {board.fixed_cells}")
        # board.print()
        tango_board, sols, check_counts = recursive_tango(position=36,
                                                        tango_board=board,
                                                        solutions=list())
        n_sols = len(sols)
        print('Solutions:', n_sols)
        print('Possible solutions checked:', check_counts)
        # if n_sols <= 5:
        #     for sol in sols:
        #         sol.print()
        sols_array[i] = n_sols
        print('*****************')
    
    avg_sols = sols_array.mean()
    min_sols = sols_array.min()
    max_sols = sols_array.max()

    print("Minimum number of solutions:", min_sols)
    print("Maximum number of solutions:", max_sols)
    print("Average number of solutions:", avg_sols)