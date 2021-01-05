from msklar3_mdiamond8_chess_helper import piece_equal, empty_path_squares, gen_state, move_to_action
import random
import numpy as np
import chess

class ParticleFilter():
  def __init__(self, board, color, N=40_000, epsilon=0.60):
    self.color = color
    self.particles = [(board.copy(), 1/N) for _ in range(N)]
    self.N = N
    self.epsilon = epsilon # chance of assuming the opponent made a random move
    self.random_weight = 1/(N+1) # the weight of random moves
  
  def update_opponent_move_result(self, captured_piece, captured_square, network):
    """
      This function is called at the start of your turn and gives you the chance to update your board.

      :param captured_piece: bool - true if your opponents captured your piece with their last move
      :param captured_square: chess.Square - position where your piece was captured
      :param network: Net - our neural network for predicting opponent move distribution
      
      Goal is to update the particles based on how we believe opponents moved and update the weights accordingly
    """
    self.particles = [self.update_particle_by_opponent_move(board, weight, captured_piece, captured_square, network) for (board, weight) in self.particles]
   
  def update_sense_result(self, sense_result):
    """
        This is a function called after you pick your 3x3 square to sense and lets us update our particles.
        

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
    """
    for i, particle in enumerate(self.particles):
      board, weight = particle
      
      new_weight = 1
      for sense in sense_result:
        square, piece = sense
        previous_piece = board.piece_at(square)
        if not piece_equal(piece, previous_piece):
          new_weight *= 0.0001
        board.set_piece_at(square, piece)
        if not board.is_valid():
          new_weight *= 0.000001
      new_weight *= weight
      self.particles[i] = (board, new_weight)
    self.update_particles_by_weight()
    self.normalize_particles()

  def update_handle_move_result(self, taken_move, captured_piece, captured_square):
    """
      This is a function called at the end of your turn/after your move was made and gives you the chance to update
      your board.

      :param taken_move: chess.Move -- the move that was actually made
      :param reason: String -- description of the result from trying to make requested_move
      :param captured_piece: bool - true if you captured your opponents piece
      :param captured_square: chess.Square - position where you captured the piece
    """
    if taken_move == None:
      return # TODO: maybe update in case a move was attempted but blocked unexpectedly
    empty_squares = empty_path_squares(taken_move) # list of squares we know are empty based on our move
    
    for i, (board, weight) in enumerate(self.particles):
      board = board.copy()
      
      castling = board.is_castling(taken_move)
      
      if captured_piece:
        if board.piece_at(captured_square) is None:
          weight *= 0.0001
      try:
        board.push(taken_move)  
      except:
        weight *= 0.0001
        
      if not castling:
        for square in empty_squares:
          if board.piece_at(square) is not None:
            board.set_piece_at(square, None)
            weight *= 0.0001
      
      self.particles[i] = (board, weight)
    self.update_particles_by_weight()
    self.normalize_particles()
      
  def sample_from_particles(self, K=1, max_iter=20):
    """
    Sample randomly from the particles based on their weights

    :param k: int -- the number of particles to sample
   
    :return: list(tuple(board, int)) -- the particle returned in the form (board, weight)
    """
    return random.choices(self.particles, weights=[weight for (board, weight) in self.particles], k=K)
    

  def update_particle_by_opponent_move(self, board, weight, captured_piece, captured_square, network):
    board = board.copy()
    board.turn = not self.color
    possible_moves = list(board.generate_pseudo_legal_moves())

    if captured_piece:
      f = lambda move: move.to_square == captured_square
      possible_moves = list(filter(f, possible_moves))    
    else:
      f = lambda move: move.to_square not in list(chess.SquareSet(board.occupied_co[self.color]))
      possible_moves = list(filter(f, possible_moves))
      
    if not possible_moves:
      weight *= 0.00001
    else:
      if random.random() < self.epsilon:
        move = np.random.choice(possible_moves)
        weight *= self.random_weight
      else:
        state, possible_board_moves = gen_state(board, not self.color)
        policy = network.PolicyForward(state, possible_board_moves).detach().numpy()
        
        possible_move_weights = np.ndarray(len(possible_moves))
        for i,move in enumerate(possible_moves):
          possible_move_weights[i] = policy[move_to_action(move)]
        
        if np.all(possible_move_weights == 0):
          move = np.random.choice(possible_moves)
          weight *= 0.00001
        else:
          move = np.random.choice(possible_moves, p=possible_move_weights / np.sum(possible_move_weights))
          weight *= policy[move_to_action(move)]
      board.push(move)
    return (board, weight)

  def update_particles_by_weight(self):
    """
    Resample each particle based on the weight
    """
    self.particles = self.sample_from_particles(self.N)
  
  def normalize_particles(self, total_weight=None):
    if total_weight is None:
      total_weight = 0
      for (board, weight) in self.particles:
        total_weight += weight
    for i, (board, weight) in enumerate(self.particles):
      self.particles[i] = (board, weight / total_weight)
      