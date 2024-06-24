import numpy as np
import copy
from typing import Any

Board = np.ndarray

class LRUDict(dict):
    def __init__(self, max_size=None):
        super().__init__()
        self.max_size = max_size
        self.order = []  # Doubly linked list to maintain order of key access

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self._update_order(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._update_order(key)

        if self.max_size is not None and len(self) > self.max_size:
            oldest_key = self.order.pop(0)
            del self[oldest_key]

    def _update_order(self, key):
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)

    def get_least_recently_used_key(self):
        return self.order[0] if self.order else None


def get_height(board: Board, pos: tuple[int, int]) -> int:
    if np.any(board[:, pos[1]] == 0):
        lowest_zero_height = np.where(board[:, pos[1]] == 0)[0][-1]
        return lowest_zero_height - pos[0]
    else:
        return -1

def check_consecutive_ones(arr, player: str = '1'):
    for row in arr:
        if player*4 in ''.join(map(str, row)):
            return True

    # Check vertically
    for col in arr.T:
        if player*4 in ''.join(map(str, col)):
            return True

    # Check diagonally (from top-left to bottom-right)
    for i in range(len(arr) - 3):
        for j in range(len(arr[0]) - 3):
            if all(arr[i+k][j+k] == int(player) for k in range(4)):
                return True

    # Check diagonally (from top-right to bottom-left)
    for i in range(len(arr) - 3):
        for j in range(3, len(arr[0])):
            if all(arr[i+k][j-k] == int(player) for k in range(4)):
                return True

    return False


def count_horiz_pairs(board: Board, player: int):
    def calculate_potential(board: Board, row: int, coli: int, counter: int) -> dict:
        d = {}
        potential_pos = []
        match counter:
            case 2: potential_pos = [-2, -1, 1, 2]
            case 3: potential_pos = [-1, 1]
        for p in potential_pos:
            new_pos = coli - counter + p
            if p > 0:
                new_pos += counter - 1
            if 0 <= new_pos < len(board[row]):
                height = get_height(board, (row, new_pos))
                if height >= 0:
                    d[p] = height
        return d

    m = {2: [], 3: [], 4: []}
    for rowi, row in enumerate(board):
        coli = 0
        counter = 0
        while coli < len(row):
            if row[coli] == player:
                counter += 1
            else:
                d = calculate_potential(board, rowi, coli, counter)
                if counter == 4 or d:
                    m[counter].append(d)
                counter = 0
            coli += 1
        d = calculate_potential(board, rowi, coli, counter)
        if counter == 4 or d:
            m[counter].append(d)
    return m


def count_diag_pairs(board: Board, player: int, flipped: bool = False):
    correct_board = np.rot90(board, 3) if flipped else board
    def calculate_potential(board: Board, row: int, coli: int, counter: int) -> dict:
        if flipped: row, coli = coli, row
        d = {}
        potential_pos = []
        match counter:
            case 2: potential_pos = [-2, -1, 1, 2]
            case 3: potential_pos = [-1, 1]
        for p in potential_pos:
            new_col_pos = coli - counter + p
            new_row_pos = row - counter + p
            if p > 0:
                new_col_pos += counter - 1
                new_row_pos += counter - 1
            if 0 <= new_col_pos < len(board[0]) and 0<= new_row_pos < len(board.T[0]):
                height = get_height(board, (new_row_pos, new_col_pos))
                if height >= 0:
                    d[p] = height
        return d
    
    m = {2: [], 3: [], 4: []}
    for top in range(len(board[0])):
        left = 0
        counter = 0
        while left < len(board) and top < len(board[0]):
            if board[left, top] == player:
                counter += 1
            else:
                d = calculate_potential(correct_board, left, top, counter)
                if counter == 4 or d:
                    m[counter].append(d)
                counter = 0
            left += 1
            top += 1
        d = calculate_potential(correct_board, left, top, counter)
        if counter == 4 or d:
            m[counter].append(d)
    for left in range(1, len(board)):
        top = 0
        counter = 0
        while left < len(board) and top < len(board[0]):
            if board[left, top] == player:
                counter += 1
            else:
                d = calculate_potential(correct_board, left, top, counter)
                if counter == 4 or d:
                    m[counter].append(d)
                counter = 0
            left += 1
            top += 1
        d = calculate_potential(correct_board, left, top, counter)
        if counter == 4 or d:
            m[counter].append(d)
    return m

def count_vert_pairs2(board: Board, player: int) -> dict[int, int]:
    d = {2: 0, 3: 0, 4: 0}
    for coli in range(len(board[0])):
        vert = board[:, coli]
        if vert[0] != 0: continue
        count = 0
        is_added = False
        for elem in vert:
            if elem != player and elem != 0:
                if count in d:
                    d[count] += 1
                    is_added = True
                break
            elif elem == player:
                count += 1
        if not is_added:
            if count in d:
                d[count] += 1
    return d

def count_vert_pairs(board: Board, player: int) -> dict[int, int]:
        to_str = lambda a: ''.join(a.astype(str))
        d = {2: 0, 3: 0, 4: 0}
        strrow = list(map(to_str, board.T))
        for i in range(2, 5):
            player_win_str = '0' + ('{0}' * i).format(player)
            for row in strrow:
                d[i] += row.count(player_win_str)
        d[2] -= d[4]
        d[3] -= d[4]
        d[4] = 0
        player_win_str = ('{0}{0}{0}{0}').format(player)
        for row in strrow:
            d[4] += row.count(player_win_str)
        return d


class AIPlayer:
    def __init__(self, player_number, weights: dict[str, Any] | None = None):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        if weights is None:
            self.weights: dict[str, Any] = {
                'hpair': (100, 500, 100000), # 2 3 4
                'vpair': (100, 500, 100000), # 2 3 4
                'dpair': (100, 500, 100000), # 2 3 4
                'hdrop': (
                    (-18, -25), # for 2 (-1/1, -2/2)
                    (-40,)      # for 3 (-1/1)
                ),
                'enemyf': 6
            }
        else: self.weights = weights

        self.cache = {
            'expandBoard': LRUDict(max_size=819200),
            'evaluation': LRUDict(max_size=819200),
        }

    def get_alpha_beta_move(self, board: Board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        expand1 = self.expand_board(board, 1 if self.player_number == 2 else 2)
        for e in expand1:
            if check_consecutive_ones(e[1], str(1 if self.player_number == 2 else 2)):
                return e[0]
        levels = 5
        return self.get_max_move(board, levels,float('inf'))[1]
    
    def expand_board(self, board: Board, player_number: int) -> list[tuple[int, Board]]:
        positions: list[tuple[int, int]] = []
        for move in range(7):
            if 0 in board[:,move]:
                update_row = -1
                for row in range(1, board.shape[0]):
                    update_row = -1
                    if board[row, move] > 0 and board[row-1, move] == 0:
                        update_row = row-1
                    elif row==board.shape[0]-1 and board[row, move] == 0:
                        update_row = row

                    if update_row >= 0:
                        positions.append((update_row, move))
                        break
        new = []
        for pos in positions:
            narr = np.array(board, copy=True)
            narr[pos] = player_number
            new.append((pos[1], narr))
        return new

    def get_min_move(self, board: Board, level: int, curr_max: float) -> tuple[float, int]:
        next_states = self.expand_board(board, 1 if self.player_number == 2 else 2)
        if len(next_states) == 0:
            return self.evaluation_function(board), -1
        if level == 0:
            return min(map(lambda x: (self.evaluation_function(x[1]), x[0]), next_states))
        
        current_min = float('inf')
        min_move = -1
        for s in next_states:
            maxmove = self.get_max_move(s[1], level - 1,current_min)
            if maxmove[0] < current_min:
                current_min = maxmove[0]
                min_move = s[0]
            if maxmove[0] < curr_max:
                return current_min, min_move

        return current_min, min_move    

    def get_max_move(self, board: Board, level: int, curr_min) -> tuple[float, int]:
        next_states = self.expand_board(board, self.player_number)
        if len(next_states) == 0:
            return self.evaluation_function(board), -1
        if level == 0:
            return max(map(lambda x: (self.evaluation_function(x[1]), x[0]), next_states))
        
        for state in next_states:
            if check_consecutive_ones(state[1], str(self.player_number)):
                return self.evaluation_function(state[1]), state[0]
        current_max = -float('inf')
        max_move = -1
        for s in next_states:
            minmove = self.get_min_move(s[1], level - 1, current_max)
            if minmove[0] > current_max:
                current_max = minmove[0]
                max_move = s[0]
            if minmove[0] > curr_min:
                return current_max, max_move
        return current_max, max_move            

    def get_chance_move(self, board: Board) -> tuple[float, int]:
        next_states = self.expand_board(board, 1 if self.player_number == 2 else 2)
        if len(next_states) == 0:
            return self.evaluation_function(board), -1
        return sum(map(lambda x: (self.evaluation_function(x[1])), next_states))//len(next_states),-1

    def get_exp_max_move(self, board: Board, level: int) -> tuple[float, int]:
        next_states = self.expand_board(board, self.player_number)

        if len(next_states) == 0:
            return self.evaluation_function(board), -1
        
        if level == 0:
            return max(map(lambda x: (self.evaluation_function(x[1]), x[0]), next_states))
        
        for state in next_states:
            if check_consecutive_ones(state[1], str(self.player_number)):
                return self.evaluation_function(state[1]), state[0]
            
        current_max = -float('inf')
        max_move = -1

        for s in next_states:
            minmove = self.get_chance_move(s[1])
            if minmove[0] > current_max:
                current_max = minmove[0]
                max_move = s[0]

        return current_max, max_move

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        expand1 = self.expand_board(board, 1 if self.player_number == 2 else 2)
        for e in expand1:
            if check_consecutive_ones(e[1], str(1 if self.player_number == 2 else 2)):
                return e[0]
        levels = 4
        return self.get_exp_max_move(board, levels)[1]



    def evaluation_function(self, board) -> float:
        if (board.tobytes()) in self.cache['evaluation']:
            return self.cache['evaluation'][(board.tobytes())]
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        answer =  self.evfn(board, self.player_number) - self.weights['enemyf'] * self.evfn(board, 1 if self.player_number == 2 else 2)
        self.cache['evaluation'][(board.tobytes())] = answer
        return answer
    
    def evfn(self, board: Board, player: int):
        horiz_stats = count_horiz_pairs(board, player)
        verti_stats = count_vert_pairs(board, player)
        diagn_stats1 = count_diag_pairs(board, player)
        rotated = np.rot90(board)
        diagn_stats2 = count_diag_pairs(rotated, player, flipped=True)

        def evaluate_stat(stats: dict[int, list[dict[int, int]]], pairw: list[int]):
            h = 0
            h2 = stats[2]
            h += len(h2) * pairw[0]
            for pairing in h2:
                for pos in pairing:
                    h += pairing[pos] * self.weights['hdrop'][0][abs(pos) - 1]
            h3 = stats[3]
            h += len(h3) * pairw[1]
            for pairing in h3:
                for pos in pairing:
                    h += pairing[pos] * self.weights['hdrop'][1][abs(pos) - 1]
            
            h4 = stats[4]
            h += len(h4) * pairw[2]
            return h
        
        h = evaluate_stat(horiz_stats, self.weights['hpair'])
        d1 = evaluate_stat(diagn_stats1, self.weights['dpair'])
        d2 = evaluate_stat(diagn_stats2, self.weights['dpair'])
        v = verti_stats[2] * self.weights['vpair'][0] + \
            verti_stats[3] * self.weights['vpair'][1] + \
            verti_stats[4] * self.weights['vpair'][2]

        return h + d1 + d2 + v


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)

class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move


