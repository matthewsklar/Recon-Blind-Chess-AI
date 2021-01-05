#!/usr/bin/env python3

"""
File Name:      game.py
Authors:        Michael Johnson and Leng Ghuy
Date:           March 18th, 2019

Description:    Python file that contains the game mechanics
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import chess
from datetime import datetime


class Game:

    def __init__(self, seconds_left=600):
        self.turn = chess.WHITE  # True for white, False for black

        self.truth_board = chess.Board()
        self.white_board = chess.Board()
        self.black_board = chess.Board()
        
        white_fen = "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        black_fen = "rnbqkbnr/pppppppp/8/8/8/8/8/8 w KQkq - 0 1"
        
        self.white_board.set_fen(white_fen)
        self.black_board.set_fen(black_fen)
        
        self.is_finished = False

        self.seconds_left_by_color = {chess.WHITE: seconds_left, chess.BLACK: seconds_left}
        self.current_turn_start_time = None

        self.move_result = None
        
    def start(self):
        """
        Starts off the clock for the first player.
        """
        self.current_turn_start_time = datetime.now()

    def end(self):
        """
        Ends the game.
        """
        self.seconds_left_by_color[self.turn] = self.get_seconds_left()
        self.is_finished = True

    def get_seconds_left(self):
        """
        :return: float -- The amount of seconds left for the current player.
        """
        if not self.is_finished and self.current_turn_start_time:
            elapsed_since_turn_start = (datetime.now() - self.current_turn_start_time).total_seconds()
            return self.seconds_left_by_color[self.turn] - elapsed_since_turn_start
        else:
            return self.seconds_left_by_color[self.turn]
        
    ###=== Generate Legal Moves ===###
    def _without_opponent_pieces(self, board, turn):
        """
        Returns a copy of the board with the opponent's pieces removed.
        :param board: chess.Board -- a chess board where you want opponnet's pieces to be removed
        :param turn: bool - True(WHITE's turn) or False(BLACK's turn), the opponnet is the 'not turn'
        
        :return: a chess.Board object
        """
        b = board.copy()
        for piece_type in chess.PIECE_TYPES:
            for sq in b.pieces(piece_type, not turn):
                b.remove_piece_at(sq)
        return b
    
    def _moves_without_opponent_pieces(self, board, turn):
        """
        Returns list of legal moves without regard to opponent piece locations.
        :param board: chess.Board -- a chess board where you want opponnet's pieces to be removed
        :param turn: bool - True(WHITE's turn) or False(BLACK's turn), the opponnet is the 'not turn'
        
        :return: List(chess.Move)
        """
        return list(self._without_opponent_pieces(board, turn).generate_pseudo_legal_moves())
    
    def _pawn_capture_moves_on(self, board, turn):
        """
        Generates all pawn captures on `board`, even if there is no piece to capture. All promotion moves are included.
        :param board: chess.Board -- a chess board where you want opponnet's pieces to be removed
        :param turn: bool - True(WHITE's turn) or False(BLACK's turn), the opponnet is the 'not turn'
        
        :return: List(chess.Move)
        """
        pawn_capture_moves = []

        no_opponents_board = self._without_opponent_pieces(board, turn)

        for pawn_square in board.pieces(chess.PAWN, turn):
            for attacked_square in board.attacks(pawn_square):
                # skip this square if one of our own pieces are on the square
                if no_opponents_board.piece_at(attacked_square):
                    continue

                pawn_capture_moves.append(chess.Move(pawn_square, attacked_square))

                # add in promotion moves
                if attacked_square in chess.SquareSet(chess.BB_BACKRANKS):
                    for piece_type in chess.PIECE_TYPES[1:-1]:
                        pawn_capture_moves.append(chess.Move(pawn_square, attacked_square, promotion=piece_type))

        return pawn_capture_moves
    
    def get_moves(self):
        """
        Returns list of legal moves without regard to opponent piece locations. Allows for pawns to move diagonally.
        :return: List(chess.Move)
        """
        if self.is_finished:
            return None
                
        return self._moves_without_opponent_pieces(self.truth_board,self.turn) + \
                self._pawn_capture_moves_on(self.truth_board, self.turn)
    
    ###=== Make move and update board ===###
    def _capture_square_of_move(self, board, move):
        """
        This function finds the the captured square if the given move captures a piece
        
        :param board: chess.Board -- a board of the current game
        :param move: chess.Move -- the move to be taken on the current board
        
        :return: chess.SQUARE -- the square where an opponent's piece is captured
                 None -- if there is no captured piece
        """
        capture_square = None
        if move is not None and board.is_capture(move):
            if board.is_en_passant(move):
                down = -8 if board.turn == chess.WHITE else 8
                capture_square = board.ep_square + down
            else:
                capture_square = move.to_square
        return capture_square
        
    def _is_psuedo_legal_castle(self, board, move):
        return board.is_castling(move) and not self._is_illegal_castle(board, move)
    
    def _is_illegal_castle(self, board, move):
        if not board.is_castling(move):
            return False

        # illegal without kingside rights
        if board.is_kingside_castling(move) and not board.has_kingside_castling_rights(board.turn):
            return True

        # illegal without queenside rights
        if board.is_queenside_castling(move) and not board.has_queenside_castling_rights(board.turn):
            return True

        # illegal if any pieces are between king & rook
        rook_square = chess.square(7 if board.is_kingside_castling(move) else 0, chess.square_rank(move.from_square))
        between_squares = chess.SquareSet(chess.BB_BETWEEN[move.from_square][rook_square])
        if any(map(lambda s: board.piece_at(s), between_squares)):
            return True

        # its legal
        return False
    
    def _slide_move(self, board, move):
        psuedo_legal_moves = list(board.generate_pseudo_legal_moves())
        squares = list(chess.SquareSet(chess.BB_BETWEEN[move.from_square][move.to_square])) + [move.to_square]
        squares = sorted(squares, key=lambda s: chess.square_distance(s, move.from_square), reverse=True)
        for slide_square in squares:
            revised = chess.Move(move.from_square, slide_square, move.promotion)
            if revised in psuedo_legal_moves:
                return revised
        return None
    
    def _add_pawn_queen_promotion(self, move):
        back_ranks = list(chess.SquareSet(chess.BB_BACKRANKS))
        piece = self.truth_board.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.PAWN and move.to_square in back_ranks and move.promotion is None:
            move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
        return move
    
    def _revise_move(self, move):
        # if its a legal move, don't change it at all. note that board.generate_psuedo_legal_moves() does not
        # include psuedo legal castles
        if move in self.truth_board.generate_pseudo_legal_moves() or self._is_psuedo_legal_castle(self.truth_board, move):
            return move

        # note: if there are pieces in the way, we DONT capture them
        if self._is_illegal_castle(self.truth_board, move):
            return None

        # if the piece is a sliding piece, slide it as far as it can go
        piece = self.truth_board.piece_at(move.from_square)
        if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
            move = self._slide_move(self.truth_board, move)

        return move if move in self.truth_board.generate_pseudo_legal_moves() else None
    
    def handle_move(self, requested_move):
        """
        Takes in the agent requested move and updatest he board accordingly with any possible rule revision
        :param requested_move: chess.Move -- the move the agent requested
        
        :return requested_move: chess.Move -- the move the agent requested
        :return taken_move: chess.Move -- the move that was actually taken 
        :return captured_square: chess.SQUARE -- the square where an opponent's piece is captured
                                 None -- if there is no captured piece
        """
        
        if self.is_finished:
            return requested_move, None, None, ""
            
        if requested_move is None:
            taken_move = None   #pass move
            captured_square = None #doesn't capture anything
            reason = "Ran out of time or None object passed in"
        elif requested_move not in self.get_moves():  #checks legality of move
            taken_move = None   #pass move
            captured_square = None #doesn't capture anything
            reason = "{} is an illegal move made.".format(requested_move)
        else:
            move = self._add_pawn_queen_promotion(requested_move)
            taken_move = self._revise_move(move)
            captured_square = self._capture_square_of_move(self.truth_board, taken_move)
            reason = ""
        
        # push move to appropriate boards for updates #
        self.truth_board.push(taken_move if taken_move is not None else chess.Move.null())
        #if self.turn == chess.WHITE: self.white_board.push(taken_move if taken_move is not None else chess.Move.null())
        #else: self.black_board.push(taken_move if taken_move is not None else chess.Move.null())
        
        if self.turn == chess.WHITE:
            self.white_board.set_fen(self._without_opponent_pieces(self.truth_board, self.turn).fen())
            self.black_board.set_fen(self._without_opponent_pieces(self.truth_board, not self.turn).fen())
        else:
            self.black_board.set_fen(self._without_opponent_pieces(self.truth_board, self.turn).fen())
            self.white_board.set_fen(self._without_opponent_pieces(self.truth_board, not self.turn).fen())
        
        # store captured_square to notify other player
        self.move_result = captured_square
        
        return requested_move, taken_move, captured_square, reason

    ###=== Handle sense square ===### 
    def handle_sense(self, square):
        """
        This function takes the sense square and returns the true state of the 3x3 section
        
        :param square: chess.SQUARES -- the square the agent wants to senese around
        :return: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                 was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        """
        if square not in list(chess.SQUARES):
            return []
        
        rank, file = chess.square_rank(square), chess.square_file(square)
        sense_result = []
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_result.append((sense_square, self.truth_board.piece_at(sense_square)))
        
        #update sense result for each respective color board
        if self.turn == chess.WHITE:
            for square, piece in sense_result:
                self.white_board.set_piece_at(square, piece)
        else:
            for square, piece in sense_result:
                self.black_board.set_piece_at(square, piece)

        return sense_result

    ###=== Return captured square ===###
    def opponent_move_result(self):
        """
        This function returns the capture square to the oppossing player
        
        :return: chess.SQUARE -- the square location where a piece was captured during the turn
        """
        return self.move_result
        
    ###=== Switch player to move ===###
    def end_turn(self):
        """
        Ends the turn for the game and updates the following
            . Updates the time used for the current player
            . Ends the turn for the current player
            . Starts the timer for the next player
        """
        
        elapsed = datetime.now() - self.current_turn_start_time
        self.seconds_left_by_color[self.turn] -= elapsed.total_seconds()

        self.turn = not self.turn
        self.current_turn_start_time = datetime.now()
        
    def is_over(self):
        """
        The function determines whether the game is over based on missing King or time_left is less than 0
        
        :return: bool -- True if the game is over, False otherwise
        """
        if self.is_finished:
            return True

        no_time_left = self.seconds_left_by_color[chess.WHITE] <= 0 or self.seconds_left_by_color[chess.BLACK] <= 0
        king_captured = self.truth_board.king(chess.WHITE) is None or self.truth_board.king(chess.BLACK) is None
        return no_time_left or king_captured
        
    def get_winner(self):
        """
        This function determines the winner color and the reason for the win
        
        :return: chess.WHITE/chess.BLACK, str -- the winning color, a string detailing the winning reason
        """
        if not self.is_over():
            return None
            
        if self.seconds_left_by_color[chess.WHITE] <= 0:
            return chess.BLACK, "BLACK won by timeout"
        elif self.seconds_left_by_color[chess.BLACK] <= 0:
            return chess.WHITE, "WHITE won by timeout"
            
        if self.truth_board.king(chess.WHITE) is None:
            return chess.BLACK, "BLACK won by king capture."
        elif self.truth_board.king(chess.BLACK) is None:
            return chess.WHITE, "WHITE won by king capture."