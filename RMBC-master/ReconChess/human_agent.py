#!/usr/bin/env python3

"""
File Name:      human_agent.py
Authors:        Michael Johnson and Leng Ghuy
Date:           03/13/2019

Description:    Python file for human agent to play in console.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
from player import Player

square_file_map = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}


class Human(Player):

    def __init__(self):
        pass

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        """
        pass
        
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        pass

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """

        s = input("----PERFORMING SENSE----\n"
                  "Enter a chess square in the form of row [A-H], column [1-8] as a single string.\n"
                  "Example: C7\n"
                  "Choice: ")
        valid = False
        while not valid:
            s = s.lower()

            if len(s) != 2:
                s = input("ERROR: length of string does not equal 2. Please try again.\n"
                          "Choice: ")
                continue

            if not s[0].isalpha() or s[0] not in chess.FILE_NAMES:
                s = input("ERROR: first index is not a letter A-H. Please try again.\n"
                          "Choice: ")
                continue

            if not s[1].isnumeric() or s[1] not in chess.RANK_NAMES:
                s = input("ERROR: second index is not a number 1-8. Please try again.\n"
                          "Choice: ")
                continue

            valid = True

        # Convert to chess.SQUARE
        file = chess.FILE_NAMES.index(s[0])
        rank = chess.RANK_NAMES.index(s[1])

        choice = chess.square(file, rank)
        return choice
        
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

        print("\nSeconds left: ", seconds_left)
        print("\nPossible moves:\n")
        for m in possible_moves:
            print(m.uci().upper(), end='  ')
        print('\n')

        s = input("----PERFORMING MOVE----\n"
                  "Enter the chess move in the form of consecutive chess squares as a single string. "
                  "Promotions made should be included as a 5th character representing the new piece type.\n"
                  "Example: Moving from C7 to C8 is 'C7C8'\n"
                  "Example: Moving from A7 to A8 with a promotion to a queen is 'A7A8Q'\n"
                  "Choice: ")
        valid = False
        while not valid:
            s = s.lower()

            if not (len(s) == 4 or len(s) == 5):
                s = input("ERROR: length of string does not equal 4 or 5. Please try again.\n"
                          "Choice: ")
                continue

            if not s[0].isalpha() or s[0] not in chess.FILE_NAMES:
                s = input("ERROR: first index is not a letter A-H. Please try again.\n"
                          "Choice: ")
                continue

            if not s[1].isnumeric() or s[1] not in chess.RANK_NAMES:
                s = input("ERROR: second index is not a number 1-8. Please try again.\n"
                          "Choice: ")
                continue

            if not s[2].isalpha() or s[2] not in chess.FILE_NAMES:
                s = input("ERROR: third index is not a letter A-H. Please try again.\n"
                          "Choice: ")
                continue

            if not s[3].isnumeric() or s[3] not in chess.RANK_NAMES:
                s = input("ERROR: fourth index is not a number 1-8. Please try again.\n"
                          "Choice: ")
                continue

            if len(s) == 5:
                if not s[4].isalpha() or s[4] not in ['q', 'r', 'b', 'n']:
                    s = input("ERROR: fifth index is not a letter representing a promotional piece "
                              "('Q', 'R', 'B', 'N'). Please try again.\n"
                              "Choice: ")
                    continue

            valid = True

        choice = chess.Move.from_uci(s)

        return choice
        
    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        pass
        
    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        pass
