#!/usr/bin/env python3

"""
File Name:      play_game.py
Authors:        Michael Johnson and Leng Ghuy
Date:           March 8th, 2019

Description:    Python file used to play a game of recon chess with two agents.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import argparse
import random
import chess
from player import load_player
from game import Game
from datetime import datetime
import time


def play_local_game(white_player, black_player, player_names):
    players = [black_player, white_player]

    game = Game()

    # writing to files
    time = "{}".format(datetime.today()).replace(" ", "_").replace(":", "-").replace(".", "-")
    filename_game = "GameHistory/" + time + "game_boards.txt"
    filename_true = "GameHistory/" + time + "true_boards.txt"
    output = open(filename_game, "w")
    output_true = open(filename_true, "w")
    output.write("Starting Game between {}-WHITE and {}-BLACK\n".format(player_names[0], player_names[1]))
    output_true.write("Starting Game between {}-WHITE and {}-BLACK\n".format(player_names[0], player_names[1]))

    white_player.handle_game_start(chess.WHITE, chess.Board())
    black_player.handle_game_start(chess.BLACK, chess.Board())
    game.start()

    move_number = 1
    while not game.is_over():
        if game.turn:
            output.write("##################################--WHITE's Turn [{}]\n".format(move_number))
            output.write("##################################--Current Board State\n")
            format_write_board(output, game.white_board)
            output_true.write("##################################--WHITE's Turn [{}]\n".format(move_number))

            print("WHITE's Turn [{}]".format(move_number))
            format_print_board(game.white_board)

        else:
            output.write("##################################--BLACK's Turn [{}]\n".format(move_number))
            output.write("##################################--Current Board State \n")
            format_write_board(output, game.black_board)
            output_true.write("##################################--BLACK's Turn [{}]\n".format(move_number))

            print("BLACK's Turn [{}]".format(move_number))
            format_print_board(game.black_board)

        output_true.write("##################################--Current Board State\n")
        format_write_board(output_true, game.truth_board)

        requested_move, taken_move = play_turn(game, players[game.turn], game.turn, move_number, output, output_true)
        print_game(game, move_number, game.turn, requested_move, taken_move)
        move_number += 1

        print("==================================\n")

    winner_color, winner_reason = game.get_winner()

    white_player.handle_game_end(winner_color, winner_reason)
    black_player.handle_game_end(winner_color, winner_reason)

    output.write("Game Over!\n")
    if winner_color is not None:
        output.write(winner_reason)
    else:
        output.write('Draw!')
    return winner_color, winner_reason


def play_turn(game, player, turn, move_number, output, output_true):
    possible_moves = game.get_moves()
    possible_sense = list(chess.SQUARES)

    # notify the player of the previous opponent's move
    captured_square = game.opponent_move_result()
    player.handle_opponent_move_result(captured_square is not None, captured_square)

    # play sense action
    sense = player.choose_sense(possible_sense, possible_moves, game.get_seconds_left())
    sense_result = game.handle_sense(sense)
    player.handle_sense_result(sense_result)
    print_sense(game, turn, sense)

    output.write("##################################--Sense Around Square {}\n".format(chess.SQUARE_NAMES[sense]))
    if turn:
        format_write_board(output, game.white_board)
    else:
        format_write_board(output, game.black_board)

    # play move action
    move = player.choose_move(possible_moves, game.get_seconds_left())
    requested_move, taken_move, captured_square, reason = game.handle_move(move)
    player.handle_move_result(requested_move, taken_move, reason, captured_square is not None,
                              captured_square)

    output.write("##################################--Move requested: {} -- Move taken: {}\n".format(requested_move, taken_move))
    output_true.write("##################################--Move requested: {} -- Move taken: {}\n\n".format(requested_move, taken_move))
    if turn:
        format_write_board(output, game.white_board)
    else:
        format_write_board(output, game.black_board)

    output.write("##################################--Truth Board State\n")
    format_write_board(output, game.truth_board)

    game.end_turn()
    return requested_move, taken_move


def print_game(game, move_number, turn, move_requested, move_taken):
    if not turn:
        print("[WHITE]-- Move requested: {} -- Move taken: {}".format(move_requested, move_taken))
        format_print_board(game.white_board)
    else:
        print("[BLACK]-- Move requested: {} -- Move taken: {}".format(move_requested, move_taken))
        format_print_board(game.black_board)


def print_sense(game, turn, sense):
    if turn:
        print("[WHITE]-- Sense Around Square {} --".format(chess.SQUARE_NAMES[sense]))
        format_print_board(game.white_board)
    else:
        print("[BLACK]-- Sense Around Square {} --".format(chess.SQUARE_NAMES[sense]))
        format_print_board(game.black_board)


def format_print_board(board):
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']
    fen = board.board_fen()

    fb = "   A   B   C   D   E   F   G   H  "
    fb += rows[0]
    ind = 1
    for f in fen:
        if f == '/':
            fb += '|' + rows[ind]
            ind += 1
        elif f.isnumeric():
            for i in range(int(f)):
                fb += '|   '
        else:
            fb += '| ' + f + ' '
    fb += '|'

    ind = 0
    for i in range(9):
        for j in range(34):
            print(fb[ind], end='')
            ind += 1
        print('\n', end='')
    print("")


def format_write_board(out, board):
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']
    fen = board.board_fen()

    fb = "   A   B   C   D   E   F   G   H  "
    fb += rows[0]
    ind = 1
    for f in fen:
        if f == '/':
            fb += '|' + rows[ind]
            ind += 1
        elif f.isnumeric():
            for i in range(int(f)):
                fb += '|   '
        else:
            fb += '| ' + f + ' '
    fb += '|'

    ind = 0
    for i in range(9):
        for j in range(34):
            out.write(fb[ind])
            ind += 1
        out.write('\n')
    out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Allows you to play against a bot. Useful for testing and debugging.')
    parser.add_argument('first_path', help='Path to first bot source file.')
    parser.add_argument('second_path', help='Path to second bot source file.')
    # parser.add_argument('--color', default='random', choices=['white', 'black', 'random'],
    #                    help='The color you want to play as.')
    args = parser.parse_args()

    name_one, constructor_one = load_player(args.first_path)
    player_one = constructor_one()
    name_two, constructor_two = load_player(args.second_path)
    player_two = constructor_two()

    players = [player_one, player_two]
    player_names = [name_one, name_two]

    if name_one == "Human":
        color = input("Play as (0)Random (1)White (2)Black: ")
        if color == '2' or (color == '0' and random.uniform(0, 1) < 0.5):
            players.reverse()
            player_names.reverse()

    win_color, win_reason = play_local_game(players[0], players[1], player_names)

    print('Game Over!')
    if win_color is not None:
        print(win_reason)
    else:
        print('Draw!')