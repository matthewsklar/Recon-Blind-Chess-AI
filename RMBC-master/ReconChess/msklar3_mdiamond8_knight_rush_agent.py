import random
import chess
from player import Player

WHITE_LEFT_KNIGHT = {chess.B1: chess.C3, chess.C3: chess.E4, chess.E4: chess.D6, chess.D6: chess.E8}
WHITE_RIGHT_KNIGHT = {chess.G1: chess.H3, chess.H3: chess.F4, chess.F4: chess.H5, chess.H5: chess.F6, chess.F6: chess.E8}

BLACK_LEFT_KNIGHT = {chess.B8: chess.C6, chess.C6: chess.B4, chess.B4: chess.D3, chess.D3: chess.E1}
BLACK_RIGHT_KNIGHT = {chess.G8: chess.F6, chess.F6: chess.H5, chess.H5: chess.F4, chess.F4: chess.D3, chess.D3: chess.E1}

class KnightRush(Player):
        
    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        """
        self.color = color
        self.left_knight = chess.B1 if color else chess.B8
        self.right_knight = chess.G1 if color else chess.G8
        self.left_policy = WHITE_LEFT_KNIGHT if color else BLACK_LEFT_KNIGHT
        self.right_policy = WHITE_RIGHT_KNIGHT if color else BLACK_RIGHT_KNIGHT
        
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        if captured_piece:
          if self.left_knight == captured_square:
            self.left_knight = None
          if self.right_knight == captured_square:
            self.right_knight = None

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        return random.choice(possible_sense)
        
    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        pass

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move

        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)

        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """          
        if self.left_knight is not None and self.left_knight in self.left_policy:
            return chess.Move(self.left_knight, self.left_policy[self.left_knight])
        
        if self.right_knight is not None and self.right_knight in self.right_policy:
            return chess.Move(self.right_knight, self.right_policy[self.right_knight])
          
        return random.choice(possible_moves)
        
    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool -- true if you captured your opponents piece
        :param captured_square: chess.Square -- position where you captured the piece
        """
        if taken_move is not None:
            if self.left_knight == taken_move.from_square:
                self.left_knight = taken_move.to_square
            if self.right_knight == taken_move.from_square:
                self.right_knight = taken_move.to_square
        
    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        pass
