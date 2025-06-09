import numpy as np
import random


class TangoBoard:
    def __init__(self, random_cells=0, random_rules=0):
        self.board = -1 * np.ones((6, 6), dtype=np.int8)
        self.fixed_cells = list()
        self.equal_rules = list()
        self.opp_rules = list()
        for _ in range(random_cells):
            self.set_cell_random()

        for _ in range(random_rules):
            self.set_rule_random()
            
    def apply_rules(self, cell):
        for rule in self.equal_rules:
            if cell in rule:
                cell_2 = self.apply_equal_rule(x=cell[0], y=cell[1])
                self.fixed_cells.append(cell_2)

        for rule in self.opp_rules:
            if cell in rule:
                cell_2 = self.apply_opp_rule(x=cell[0], y=cell[1])
                self.fixed_cells.append(cell_2)
    
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

    def set_rule_random(self):
        check = False
        while not check:
            possible_cells = [divmod(n, 6) for n in range(36)]
            fixed_cells = self.fixed_cells.copy()
            for rule in self.equal_rules + self.opp_rules:
                cell_1 = rule[0]
                cell_2 = rule[1]
                possible_cells.remove(cell_1)
                possible_cells.remove(cell_2)
                fixed_cells.append(cell_1)
                fixed_cells.append(cell_2)
            x, y = random.choice(possible_cells)
            rule_type = random.choice(['equal', 'opposite'])
            direction = random.choice([[0, 1], [1, 0]])
            x_2 = x + direction[0]
            y_2 = y + direction[1]
            
            if x_2 > 5 or y_2 > 5:
                check = False
            elif not (x_2, y_2) in fixed_cells:
                check = True
        
        new_fixed_cell = None
        if rule_type == 'equal':
            self.set_equal_rule(x, y, x_2, y_2)
            if (x, y) in self.fixed_cells:
                new_fixed_cell = self.apply_equal_rule(x, y)
        else:
            self.set_opp_rule(x, y, x_2, y_2)
            if (x, y) in self.fixed_cells:
                new_fixed_cell = self.apply_equal_rule(x, y)

        
        if new_fixed_cell:
            print(f"New fixed cell due to rules: {new_fixed_cell}")

        self.fixed_cells.append((x_2, y_2))

    def get_cell(self, x, y):
        return self.board[x, y]

    def set_cell(self, x, y, value):
        if not (x, y) in self.fixed_cells:
            self.board[x, y] = value
        else:
            raise ValueError(f"Trying to set a fixed cell")

    @staticmethod
    def check_contiguos_cells(x_1, y_1, x_2, y_2):
        to_raise = False
        if x_1 == x_2:
            if y_1 != y_2 + 1 and y_1 != y_2 - 1:
                to_raise = True
        elif y_1 == y_2:
            if x_1 != x_2 + 1 and x_1 != x_2 - 1:
                to_raise = True
        else:
            to_raise = True
        
        if to_raise:    
            print((x_1, y_1), (x_2, y_2))
            raise ValueError('Cells are not contiguos')

    def set_equal_rule(self, x_1, y_1, x_2, y_2):
        self.check_contiguos_cells(x_1, y_1, x_2, y_2)
        self.equal_rules.append([(x_1, y_1), (x_2, y_2)])

    def set_opp_rule(self, x_1, y_1, x_2, y_2):
        self.check_contiguos_cells(x_1, y_1, x_2, y_2)
        self.opp_rules.append([(x_1, y_1), (x_2, y_2)])
    
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
                value = self.board[i, j]
                print_value = f"{value} "
                if value != -1:
                    print_value = " " + print_value
                board += print_value
            board += " |\n"
        print(board)
        print('---------------')

    def apply_equal_rule(self, x, y):
        cell_2 = None
        if self.equal_rules:
            cell = (x, y)
            value = self.get_cell(x, y)
            for eq_rule in self.equal_rules:
                if cell in eq_rule:
                    tmp_rule = eq_rule.copy()
                    tmp_rule.remove(cell)
                    cell_2 = tmp_rule[0]
                    self.board[cell_2[0], cell_2[1]] = value
        return cell_2

    def apply_opp_rule(self, x, y):
        cell_2 = None
        if self.opp_rules:
            cell = (x, y)
            value = 1 if self.get_cell(x, y) == 0 else 0
            for opp_rule in self.opp_rules:
                if cell in opp_rule:
                    tmp_rule = opp_rule.copy()
                    tmp_rule.remove(cell)
                    cell_2 = tmp_rule[0]
                    self.board[cell_2[0], cell_2[1]] = value
        return cell_2

    def get_cell_rule(self, x, y):
        result = dict()
        cell = (x, y)
        for opp_rule in self.opp_rules:
            if cell in opp_rule:
                cell_opp = opp_rule.copy()
                cell_opp.remove(cell)
                result['opposite'] = cell_opp
                break
        for eq_rule in self.equal_rules:
            if cell in eq_rule:
                cell_eq = eq_rule.copy()
                cell_eq.remove(cell)
                result['equal'] = cell_eq
                break
        return result

    def check_equal_rules(self):
        check = True
        for rule in self.equal_rules:
            cell_1 = rule[0]
            value_1 = self.get_cell(cell_1[0], cell_1[1])
            cell_2 = rule[1]
            value_2 = self.get_cell(cell_2[0], cell_2[1])
            if value_1 != value_2:
                check = False
        return check

    def check_opp_rules(self):
        check = True
        for rule in self.opp_rules:
            cell_1 = rule[0]
            value_1 = self.get_cell(cell_1[0], cell_1[1])
            cell_2 = rule[1]
            value_2 = self.get_cell(cell_2[0], cell_2[1])
            if value_1 == value_2:
                check = False
        return check

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
        num = 1
        for cell in self.board[i, 1:]:
            if prev_cell == cell:
                num += 1
                if num == 3:
                    condition = False
                    break
            else:
                num = 1
            prev_cell = cell

        return condition
    
    def check_cons_col(self, j):
        condition = True
        prev_cell = self.board[0, j]
        num = 1
        for cell in self.board[1:, j]:
            if prev_cell == cell:
                num += 1
                if num == 3:
                    condition = False
                    break
            else:
                num = 1
            prev_cell = cell
        return condition

    def fulfill(self, board=None, eq_rules=list(), opp_rules=list()):
        if not board is None:
            self.board = np.array(board, dtype=np.int8)
            for row in range(6):
                for col in range(6):
                    if self.board[row, col] != -1:
                        self.fixed_cells.append((row, col))
        else:
            n_cells = int(input("Number of fixed cells: "))
            for c in range(n_cells):
                cell = input("Cell: ")
                cell = eval(cell)
                value = input(f"Value in position {cell}: ")
                self.set_cell(cell[0], cell[1], int(value))
                self.fixed_cells.append(cell)
            print()
        if not board is None:
            self.equal_rules = eq_rules
        else:
            n_eq_rules = input("Number of equal rules: ")
            for n in range(int(n_eq_rules)):
                cell = input("Cell 1 of equal rule: ")
                cell_1 = eval(cell)
                cell = input("Cell 2 of equal rule: ")
                cell_2 = eval(cell)
                self.equal_rules.append([cell_1, cell_2])
                print('Rule saved!\n')

        if not board is None:
            self.opp_rules = opp_rules
        else:
            n_opp_rules = input("Number of opposite rules: ")
            for n in range(int(n_opp_rules)):
                cell = input("Cell 1 of opposite rule: ")
                cell_1 = eval(cell)
                cell = input("Cell 2 of opposite rule: ")
                cell_2 = eval(cell)
                self.opp_rules.append([cell_1, cell_2])
                print('Rule saved!\n')

        self.get_fixed_cells_from_rules()

    def get_fixed_cells_from_rules(self):
        fixed_cells = self.fixed_cells.copy()

        new_cells = list()
        for x, y in fixed_cells:
            cell = self.apply_equal_rule(x, y)
            if cell:
                if cell not in self.fixed_cells:
                    self.fixed_cells.append(cell)
                    new_cells.append(cell)
            else:
                cell = self.apply_opp_rule(x, y)
                if cell:
                    if cell not in self.fixed_cells:
                        self.fixed_cells.append(cell)
                        new_cells.append(cell)
        for new_cell in new_cells:
            print(f"New fixed cells due to rules: {new_cell}")


    def check(self):
        if not self.check_opp_rules() or not self.check_equal_rules():
            return False
        
        for n in range(6):
            if not self.check_num_row(n) or not self.check_cons_row(n):
                return False
            if not self.check_num_col(n) or not self.check_cons_col(n):
                return False

        return True
