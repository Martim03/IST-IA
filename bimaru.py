# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 007:
# 103432 Lourenco Matos
# 102932 Martim Mendes

import sys

import numpy as np
import copy

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    uniform_cost_search,
    iterative_deepening_search
)

Battleship = 4
Cruiser = 3
Destroyer = 2
Submarine = 1


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        if isinstance(other, BimaruState):
            return self.id == other.id and self.board == other.board
        return False

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"BimaruState: {self.id}, Board: \n{str(self.board)}"


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, total_hints: int, hints: list, row: list, col: list):
        self.matrix = np.zeros((10, 10), dtype=object)
        self.row_values = row
        self.col_values = col
        self.ships = {Battleship: 1, Cruiser: 2, Destroyer: 3, Submarine: 4}
        self.hints = hints
        self.num_hints_placed = total_hints
        self.hints_placed = []
        self.free_positions = 100
        self.pieces_left = 20 # 1*4 + 2*3 + 3*2 + 4*1

    def handle_hints(self):
        for (x, y), piece in self.hints:
            if piece == "C":
                self.insert_circle(x, y)
                self.num_hints_placed -= 1
                self.hints_placed.append((x, y))
            elif piece == "W":
                self.insert_water(x, y)
                self.num_hints_placed -= 1
                self.hints_placed.append((x, y))
            elif piece == "T":
                if x == 8 or self.col_values[y] == 2:
                    self.insert_ship_vertical(x, y, 2)
                else:
                    self.insert_top_waters(x, y)
            elif piece == "B":
                if x == 1 or self.col_values[y] == 2:
                    self.insert_ship_vertical(x - 1, y, 2)
                else:
                    self.insert_bottom_waters(x, y)
            elif piece == "L":
                if y == 8 or self.row_values[x] == 2:
                    self.insert_ship_horizontal(x, y, 2)
                else:
                    self.insert_left_waters(x, y)
            elif piece == "R":
                if y == 1 or self.row_values[x] == 2:
                    self.insert_ship_horizontal(x, y - 1, 2)
                else:
                    self.insert_right_waters(x, y)
            elif piece == "M":
                if x == 0 and self.row_values[x] == 3:
                    self.insert_ship_horizontal(x, y - 1, 3)
                elif x == 9 and self.row_values[x] == 3:
                    self.insert_ship_horizontal(x, y - 1, 3)
                elif y == 0 and self.col_values[y] == 3:
                    self.insert_ship_vertical(x - 1, y, 3)
                elif y == 9 and self.col_values[y] == 3:
                    self.insert_ship_vertical(x - 1, y, 3)
                else:
                    self.insert_middle_waters(x, y)
            self.fill_waters()


    def __str__(self):
        result = ""
        for row in range(10):
            for col in range(10):
                if self.matrix[row][col] == 0:
                    # should not happen at the end
                    result += "0"
                    continue
                elif self.matrix[row][col] == "w":
                    if ((row, col), "W") in self.hints:
                        result += "W"
                    else:
                        result += "."
                    continue
                piece = self.matrix[row, col]
                if ((row, col), piece.upper()) in self.hints:
                    result += str(piece).upper()
                else:
                    result += str(piece)
            result += "\n"
        return result[:-1]

    ############################################ INSERT PIECES #######################################################

    def insert_water(self, row: int, col: int):
        """Insere agua na respetiva posicao"""
        if self.matrix[row,col] != "w":
            self.matrix[row, col] = "w"
            self.free_positions -= 1

    def fill_waters(self):
        """Preenche as rows e cols vazias com agua"""
        for idx in range(10):
            if self.row_values[idx] == 0:
                self.fill_water_row(idx)
            if self.col_values[idx] == 0:
                self.fill_water_col(idx)

    def fill_water_row(self, row: int):
        """Preenche uma row com agua"""
        for col in range(10):
            if self.is_free_position(row, col):
                self.insert_water(row, col)

    def fill_water_col(self, col: int):
        """Preenche uma col com agua"""
        for row in range(10):
            if self.is_free_position(row, col):
                self.insert_water(row, col)

    def insert_top_waters(self, row: int, col: int):
        adjacent_coords = self.get_all_adjacent_coords(row, col)

        adjacent_coords.remove((row + 1, col))

        for coord in adjacent_coords:
            self.insert_water(coord[0], coord[1])

        if self.is_free_position(row + 2, col - 1):
            self.insert_water(row + 2, col - 1)

        if self.is_free_position(row + 2, col + 1):
            self.insert_water(row + 2, col + 1)

        if self.col_values[col] == 1:
            self.fill_water_col(col)
            self.matrix[row, col] = 0
            self.free_positions += 1

        if self.row_values[row] == 1:
            self.fill_water_row(row)
            self.matrix[row, col] = 0
            self.free_positions += 1

    def insert_bottom_waters(self, row: int, col: int):
        adjacent_coords = self.get_all_adjacent_coords(row, col)

        adjacent_coords.remove((row - 1, col))

        for coord in adjacent_coords:
            self.insert_water(coord[0], coord[1])

        if self.is_free_position(row - 2, col - 1):
            self.insert_water(row - 2, col - 1)

        if self.is_free_position(row - 2, col + 1):
            self.insert_water(row - 2, col + 1)

        if self.col_values[col] == 1:
            self.fill_water_col(col)
            self.matrix[row, col] = 0
            self.free_positions += 1

        if self.row_values[row] == 1:
            self.fill_water_row(row)
            self.matrix[row, col] = 0
            self.free_positions += 1

    def insert_left_waters(self, row: int, col: int):
        adjacent_coords = self.get_all_adjacent_coords(row, col)

        adjacent_coords.remove((row, col + 1))

        for coord in adjacent_coords:
            self.insert_water(coord[0], coord[1])

        if self.is_free_position(row - 1, col + 2):
            self.insert_water(row - 1, col + 2)

        if self.is_free_position(row + 1, col + 2):
            self.insert_water(row + 1, col + 2)

        if self.col_values[col] == 1:
            self.fill_water_col(col)
            self.matrix[row, col] = 0
            self.free_positions += 1

        if self.row_values[row] == 1:
            self.fill_water_row(row)
            self.matrix[row, col] = 0
            self.free_positions += 1

    def insert_right_waters(self, row: int, col: int):
        adjacent_coords = self.get_all_adjacent_coords(row, col)

        adjacent_coords.remove((row, col - 1))

        for coord in adjacent_coords:
            self.insert_water(coord[0], coord[1])

        if self.is_free_position(row - 1, col - 2):
            self.insert_water(row - 1, col - 2)

        if self.is_free_position(row + 1, col - 2):
            self.insert_water(row + 1, col - 2)

        if self.col_values[col] == 1:
            self.fill_water_col(col)
            self.matrix[row, col] = 0
            self.free_positions += 1

        if self.row_values[row] == 1:
            self.fill_water_row(row)
            self.matrix[row, col] = 0
            self.free_positions += 1

    def insert_middle_waters(self, row: int, col: int):
        diagonals = [
            (row - 1, col - 1),
            (row - 1, col + 1),
            (row + 1, col - 1),
            (row + 1, col + 1),
        ]

        for coord in diagonals:
            if self.is_free_position(coord[0], coord[1]):
                self.insert_water(coord[0], coord[1])

        if self.col_values[col] == 1:
            self.fill_water_col(col)
            self.matrix[row, col] = 0
            self.free_positions += 1

        if self.row_values[row] == 1:
            self.fill_water_row(row)
            self.matrix[row, col] = 0
            self.free_positions += 1

    def insert_circle(self, row: int, col: int):
        """Insere um circulo na respetiva posicao"""

        self.matrix[row][col] = "c"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.ships[Submarine] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        adjacent_positions = self.get_all_adjacent_coords(row, col)

        for position in adjacent_positions:
            self.insert_water(position[0], position[1])

        return True

    def insert_top_piece(self, row: int, col: int):
        """Insere a peca superior na respetiva posicao"""

        self.matrix[row][col] = "t"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        waters = self.get_all_adjacent_coords(row, col)
        if (row + 1, col) in waters:
            waters.remove((row + 1, col))

        for water in waters:
            self.insert_water(water[0], water[1])

        return True

    def insert_bottom_piece(self, row: int, col: int):
        """Insere a peca inferior na respetiva posicao"""

        self.matrix[row][col] = "b"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        waters = self.get_all_adjacent_coords(row, col)
        if (row - 1, col) in waters:
            waters.remove((row - 1, col))

        for water in waters:
            self.insert_water(water[0], water[1])

        return True

    def insert_left_piece(self, row: int, col: int):
        """Insere a peca esquerda na respetiva posicao"""

        self.matrix[row][col] = "l"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        waters = self.get_all_adjacent_coords(row, col)
        if (row, col + 1) in waters:
            waters.remove((row, col + 1))

        for water in waters:
            self.insert_water(water[0], water[1])

        return True

    def insert_right_piece(self, row: int, col: int):
        """Insere a peca direita na respetiva posicao"""

        self.matrix[row][col] = "r"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        waters = self.get_all_adjacent_coords(row, col)
        if (row, col - 1) in waters:
            waters.remove((row, col - 1))

        for water in waters:
            self.insert_water(water[0], water[1])

        return True

    def insert_middle_piece(self, row: int, col: int):
        """Insere a peca do meio na respetiva posicao"""

        self.matrix[row][col] = "m"
        self.row_values[row] -= 1
        self.col_values[col] -= 1
        self.free_positions -= 1
        self.pieces_left -= 1

        waters = self.get_all_adjacent_coords(row, col)
        if (row + 1, col) in waters:
            waters.remove((row + 1, col))
        if (row - 1, col) in waters:
            waters.remove((row - 1, col))
        if (row, col + 1) in waters:
            waters.remove((row, col + 1))
        if (row, col - 1) in waters:
            waters.remove((row, col - 1))

        for water in waters:
            self.insert_water(water[0], water[1])

        return True

    def insert_value(self, row: int, col: int, value: str):
        """Insere o valor na respetiva posicao - retorna False caso nao seja possivel"""
        if value == "c":
            self.insert_circle(row, col)
        elif value == "t":
            self.insert_top_piece(row, col)
        elif value == "b":
            self.insert_bottom_piece(row, col)
        elif value == "l":
            self.insert_left_piece(row, col)
        elif value == "r":
            self.insert_right_piece(row, col)
        elif value == "m":
            self.insert_middle_piece(row, col)
        elif value == "w":
            self.insert_water(row, col)

        self.fill_waters()

    ########################################### INSERT SHIPS ##########################################################

    def insert_ship_horizontal(self, row: int, col: int, length: int):
        """Insert a ship of the given length at the given location, horizontally."""
        if length == 1:
            self.insert_circle(row, col)
        elif length == 2:
            self.insert_left_piece(row, col)
            self.insert_right_piece(row, col + 1)
            self.ships[Destroyer] -= 1
        elif length == 3:
            self.insert_left_piece(row, col)
            self.insert_middle_piece(row, col + 1)
            self.insert_right_piece(row, col + 2)
            self.ships[Cruiser] -= 1
        elif length == 4:
            self.insert_left_piece(row, col)
            self.insert_middle_piece(row, col + 1)
            self.insert_middle_piece(row, col + 2)
            self.insert_right_piece(row, col + 3)
            self.ships[Battleship] -= 1

    def insert_ship_vertical(self, row: int, col: int, length: int):
        """Insert a ship of the given length at the given location, vertically."""
        if length == 1:
            self.insert_circle(row, col)
        elif length == 2:
            self.insert_top_piece(row, col)
            self.insert_bottom_piece(row + 1, col)
            self.ships[Destroyer] -= 1
        elif length == 3:
            self.insert_top_piece(row, col)
            self.insert_middle_piece(row + 1, col)
            self.insert_bottom_piece(row + 2, col)
            self.ships[Cruiser] -= 1
        elif length == 4:
            self.insert_top_piece(row, col)
            self.insert_middle_piece(row + 1, col)
            self.insert_middle_piece(row + 2, col)
            self.insert_bottom_piece(row + 3, col)
            self.ships[Battleship] -= 1


    ########################### CHECKS WHAT SHIP LENGTH CAN BE PLACED #################################################
    def get_max_ship_length_horizontal(self, row: int, col: int) -> int:
        """Return the maximum possible length of a ship that can be placed at the given location, horizontally."""
        max_length = 0
        limit = max(size for size, count in self.ships.items() if count > 0)
        for offset in range(limit):
            if not self.is_free_position(row, col + offset) or self.col_values[col + offset] < 1:
                break
            if any(value != 'OUT' and value != 0 and value != "w"
                   for value in self.get_all_adjacent_values(row, col + offset)):  # if there's a ship nearby
                break
            max_length += 1
        return min(max_length, self.row_values[row])

    def get_max_ship_length_vertical(self, row: int, col: int) -> int:
        """Return the maximum possible length of a ship that can be placed at the given location, vertically."""
        max_length = 0
        limit = max(size for size, count in self.ships.items() if count > 0)
        for offset in range(4):
            if not self.is_free_position(row + offset, col) or self.row_values[row + offset] < 1:
                break

            if any(value != 'OUT' and value != 0 and value != "w"
                   for value in self.get_all_adjacent_values(row + offset, col)):  # if there's a ship nearby
                break
            max_length += 1
        return min(max_length, self.col_values[col])


    def can_place_vertical_ship(self, row: int, col: int, length: int) -> bool:

        if length > self.col_values[col]:
            return False

        for offset in range(length):

            if not self.is_free_position(row + offset, col) or self.row_values[row + offset] < 1:
                return False

            if any(value != 0 and value != "w" for value in self.get_all_adjacent_values(row + offset, col)):
                return False

        return True

    def can_place_horizontal_ship(self, row: int, col: int, length: int) -> bool:

        if length > self.row_values[row]:
            return False

        for offset in range(length):

            if not self.is_free_position(row, col + offset) or self.col_values[col + offset] < 1:
                return False

            if any(value != 0 and value != "w" for value in self.get_all_adjacent_values(row, col + offset)):
                return False
        return True

    ########################################### GET VALUES ###########################################################

    def is_free_position(self, row: int, col: int) -> bool:
        """Verifica se a posicao esta livre"""
        return self.get_value(row, col) == 0

    def get_value(self, row: int, col: int):
        """
        Devolve o valor na respetiva posição do tabuleiro.
        Returns -1, None or a string value.
        """
        if row < 0 or col < 0 or row >= 10 or col >= 10:
            return -1  # OUT OF BOUNDS

        return self.matrix[row, col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        verticals = [self.get_value(row - 1, col), self.get_value(row + 1, col)]

        return [x for x in verticals if x != -1]

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        horizontals = [self.get_value(row, col - 1), self.get_value(row, col + 1)]

        return [x for x in horizontals if x != -1]

    def adjacent_diagonal_values(self, row: int, col: int) -> (str, str, str, str):
        """Devolve os valores das 4 diagonais possiveis"""
        up_left = self.get_value(row - 1, col - 1)
        up_right = self.get_value(row - 1, col + 1)
        down_left = self.get_value(row + 1, col - 1)
        down_right = self.get_value(row + 1, col + 1)

        return up_left, up_right, down_left, down_right

    def get_all_adjacent_values(self, row: int, col: int) -> list:
        """Return all adjacent values."""
        # Get vertical, horizontal, and diagonal adjacent values

        # Combine all adjacent values into a single list
        adjacent_values = self.get_all_adjacent_coords(row, col)

        # Filter out any 'OUT OF BOUNDS' values
        adjacent_values = [self.matrix[x, y] for (x, y) in adjacent_values]

        return adjacent_values

    def get_all_adjacent_coords(self, row: int, col: int) -> list:
        """Return all adjacent coordinates."""
        # Get vertical, horizontal, and diagonal adjacent values
        adjacent_coords = [
            (row - 1, col),  # up
            (row + 1, col),  # down
            (row, col - 1),  # left
            (row, col + 1),  # right
            (row - 1, col - 1),  # up left
            (row - 1, col + 1),  # up right
            (row + 1, col - 1),  # down left
            (row + 1, col + 1),  # down right
        ]

        # Filter out any 'OUT OF BOUNDS' values

        adjacent_coords = [coord for coord in adjacent_coords if self.get_value(coord[0], coord[1]) != -1]

        return adjacent_coords

    ########################################### PARSE INPUT ###########################################################
    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board. """

        HINTS = []
        ROW = []
        COLUMN = []
        total_hints = 0
        for line in sys.stdin:
            if "ROW" in line:
                ROW = [int(x) for x in line.split()[1:]]
            elif "COLUMN" in line:
                COLUMN = [int(x) for x in line.split()[1:]]
            elif "HINT" in line:
                line = line.split()
                HINTS.append(((int(line[1]), int(line[2])), line[3]))
            else:
                total_hints = int(line[0])

        board = Board(total_hints, HINTS, ROW, COLUMN)
        board.handle_hints()
        board.fill_waters()
        return board

    ########################################### CHECK BOARD VALIDITY ##################################################

    def is_complete(self):
        """Check whether the board is complete."""
        if any(x for x in self.ships.values()):
            return False
        return True

    def all_hints_placed(self):
        for (row, col), piece in self.hints:
            if (self.matrix[row, col]).upper() != piece:
                return False
            if (row, col) not in self.hints_placed and self.matrix[row, col] == piece:
                self.num_hints_placed -= 1
                self.hints_placed.append((row, col))
        return True

    def find_largest_vertical_ship(self, row: int, col: int) -> int:
        """Return the size of the largest vertical ship that contains the given cell."""
        size = 1
        while row + size < 10 and self.matrix[row + size, col] == "m":
            size += 1
        if self.matrix[row + size, col] == "b":
            size += 1
        else:
            return -1

        return size

    def find_largest_horizontal_ship(self, row: int, col: int) -> int:
        """Return the size of the largest horizontal ship that contains the given cell."""
        size = 1
        while col + size < 10 and self.matrix[row, col + size] == "m":
            size += 1
        if self.matrix[row, col + size] == "r":
            size += 1
        else:
            return -1

        return size



class Bimaru(Problem):

    def __init__(self, initial: BimaruState):
        """O construtor especifica o estado inicial."""
        super().__init__(initial)
        self.board = initial.board
        # self.goal = ...

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        ships = state.board.ships

        if state.board.free_positions < state.board.pieces_left:
            return actions

        try:
            max_ship_length = max(size for size, count in ships.items() if count > 0)
        except ValueError:
            return actions

        if all(col_val < max_ship_length for col_val in state.board.col_values) and \
                all(row_val < max_ship_length for row_val in state.board.row_values):
            return actions

        # Loop over each cell in the board
        for row in range(10):
            # free positions dessa row < row_value
            if sum(state.board.is_free_position(row, y) for y in range(10)) < state.board.row_values[row]:
                return []

            if state.board.row_values[row] == 0:
                continue

            for col in range(10):

                # free positions dessa col < col_value
                if row == 0 and sum(state.board.is_free_position(x, col) for x in range(10)) < state.board.col_values[col]:
                    return []

                if state.board.col_values[col] == 0:
                    continue

                # Check if the current cell is empty
                if state.board.is_free_position(row, col):  # checks if value == 0


                    if max_ship_length == 1:
                        adjacents = state.board.get_all_adjacent_values(row, col)
                        if all(x == 0 or x == "w" for x in adjacents):
                            actions.append(("VERTICAL", (row, col), max_ship_length))

                    else:

                        if state.board.can_place_vertical_ship(row, col, max_ship_length):
                            actions.append(("VERTICAL", (row, col), max_ship_length))

                        if state.board.can_place_horizontal_ship(row, col, max_ship_length):
                            actions.append(("HORIZONTAL", (row, col), max_ship_length))
        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        orientation = action[0]
        row, col = action[1]
        length = action[2]

        new_board = copy.deepcopy(state.board)

        if orientation == "VERTICAL":
            new_board.insert_ship_vertical(row, col, length)
        elif orientation == "HORIZONTAL":
            new_board.insert_ship_horizontal(row, col, length)

        new_board.fill_waters()
        new_state = BimaruState(new_board)
        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # Assumimos que foi tudo inserido corretamente
        # ie. nao ha barcos adjacentes, etc
        #print(f"Testing: {state.id}")

        # and state.board.all_hints_placed()
        # Not needed since deduced waters are limiting solution

        return state.board.is_complete() and state.board.all_hints_placed()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # Primeira heuristica: numero de barcos por colocar
        # Prioritiza os estados com menos barcos para colocar
        state = node.state

        remaining_pieces = state.board.pieces_left + state.board.free_positions + sum(state.board.row_values) + sum(state.board.col_values) + (state.board.num_hints_placed * 10)
        return remaining_pieces



""" __NOVA FUNÇOES__ """

if __name__ == "__main__":

    # Read the board from stdin
    board = Board.parse_instance()

    # Create the initial state
    initial_state = BimaruState(board)

    # Create the Bimaru problem
    problem = Bimaru(initial_state)
    # Solve the problem
    solution_node = depth_first_tree_search(problem)

    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    if solution_node is None:
        print("No solution found!")
    else:
        print(solution_node.state.board)
