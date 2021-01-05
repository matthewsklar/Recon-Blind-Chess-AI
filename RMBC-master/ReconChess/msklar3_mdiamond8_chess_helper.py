import numpy as np
import chess
import torch

piece_map = {
    'P' : 1,
    'R' : 2,
    'N' : 3,
    'B' : 4,
    'Q' : 5,
    'K' : 6,
    'p' : 7,
    'r' : 8,
    'n' : 9,
    'b' : 10,
    'q' : 11,
    'k' : 12,
}

board_map = {
    1 : 'P',
    2 : 'R',
    3 : 'N',
    4 : 'B',
    5 : 'Q',
    6 : 'K',
    7 : 'p',
    8 : 'r',
    9 : 'n',
    10 : 'b',
    11 : 'q',
    12 : 'k',
}

def piece_equal(piece1, piece2):
  if piece1 is None:
    return piece2 is None
  if piece2 is None:
    return False
  return piece1.symbol() == piece2.symbol()

def empty_path_squares(move):
  """
  Returns the known empty squares resulting from a successful move, used in filtering.
  
  :param move: chess.Move -- the move taken, or None
  :return: list(chess.Square) -- the known empty squares based on this move
  """
  if move is None or move.from_square is None or move.to_square is None:
    return []
  if chess.square_distance(move.from_square, move.to_square) <= 1:
    return []
  x = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
  y = chess.square_file(move.to_square) - chess.square_file(move.from_square)
  if x != 0 and y != 0 and abs(x) != abs(y): # knight move check
    return []
  squares = between(int(move.from_square), int(move.to_square))
  if squares is None:
    return []
  empty_squares = chess.SquareSet(squares)
  empty_squares.add(move.from_square)
  return list(empty_squares)

def gen_state(board, color):
  """
  Generate the representation of the state for a neural network
  
  :param board: chess.Board -- the board representation
  :param color: chess.Color (boolean) -- the current turn perspective

  :return: (nn_state, possible_moves) - the network compatible state and the possible moves
  
  NOTE: this flips the board if the turn is black, this means that moves and distribution output are also FLIPPED
  """
  if not color:
    board = board.mirror() # always pretend to be on white for training time, flip the board if current move is black
  board.turn = chess.WHITE
  state = np.zeros((8,8,2,6))
  for (x,y), piece in np.ndenumerate(fen_to_board(board)):
    if piece > 0:
      state[x, y, int(piece > 6), int((piece-1) % 6)] = 1
  
  nn_state = torch.tensor(state)
  possible_moves = possible_moves_to_action_map(list(board.generate_pseudo_legal_moves()))
  return (nn_state, possible_moves)

def fen_to_board(board):
    fen = board.fen()
    board = np.zeros((8,8))

    x = 0
    y = 0

    for c in fen:
        if c == ' ':
            break
        
        if c.isdigit():
            x += int(c)
        elif c == '/':
            y += 1
            x = 0
        else:
            board[y][x] = piece_map[c]
            x += 1
    return board

def board_to_fen(board_state, color):
  fen = ""

  board = board_state[0][0]

  for r in board:
    empty_count = 0

    for f in r:
      if f == 0:
        empty_count += 1
      else:
        if empty_count != 0:
          fen += str(empty_count)
          empty_count = 0

        fen += board_map[int(f)]

    if empty_count != 0:
      fen += str(empty_count)
      empty_count = 0
      
    fen += '/'

  fen = fen[0:-1]
  fen += ' {} - - 0 1'.format('w' if color == chess.WHITE else 'b')
  print(fen)
  return fen

'''
Create 1 hot array of all possible moves
'''
def possible_moves_to_action_map(possible_moves):
  possible_moves_uci = np.zeros(64*64, dtype=int)

  for move in possible_moves:
    action_index = move_to_action(move)

    possible_moves_uci[action_index] = 1
  return possible_moves_uci

   
def mirror_square(sq):
  return int(7 - (sq % 8) + 8 * (7 - int(sq/8)))
   
def reverse_move(move): # reverse a chess.Move
  return chess.Move(mirror_square(move.from_square), mirror_square(move.to_square))

''' 
Map an action (index in the policy) to the actual move encoded as a tuple of 
chess.Move objects as (from_square, to_square).
'''
def action_map(action_id):
  from_square_id = action_id % 64
  to_square_id = int(np.floor(action_id / 64))
  
  return chess.Move(from_square_id, to_square_id)

def move_to_action(move):
  return move.from_square + move.to_square * 64
  
def between(a, b):
  try:
    bb = chess.BB_RAYS[a][b] & ((chess.BB_ALL << a) ^ (chess.BB_ALL << b))
    return bb & (bb - 1)
  except:
    print("DEBUG between(a,b) (type(a), type(b), a, b)", type(a), type(b), a, b)
    return []

def best_move_to_action_map(move):
  action_map = np.zeros((64*64,), dtype=np.double)
  action = move_to_action(move)

  action_map[action] = 1

  return action_map

def cp_to_win_probability(cp):
  if cp == 0:
    return 0

  if cp > 10:
    return 1

  if cp < -10:
    return -1  
  
  return 1 / (1 + 10**(-((cp / 10) / 4)))
