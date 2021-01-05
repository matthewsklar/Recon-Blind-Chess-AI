
#!/usr/bin/env python3

"""
File Name:      chessboard_gui.py
Authors:        Michael Johnson and Leng Ghuy
Date:           April 11th, 2019
Description:    Python file used to show a game of recon chess. Can be played in real-time or for replays.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import argparse
import random
import chess
import tkinter as tk
import tkinter.font as tkFont
import time

class ChessboardGUI():

    def __init__(self, frame_width=810, frame_height=900, names=None):
        self.square_size = int(frame_height / 10)
        self.half_square = int(self.square_size/2)

        # Window
        self.win = tk.Tk()
        self.win.title("Recon Chess Playback")

        # Calculate x and y coordinates for Tk root window
        ws = self.win.winfo_screenwidth()  # width of the screen
        hs = self.win.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (frame_width / 2)
        y = (hs / 2) - (frame_height / 2)

        self.win.geometry('%dx%d+%d+%d' % (frame_width-self.half_square, frame_height-self.half_square, x, y))

        # Colors
        self.WHITE = '#FFFFFF'
        self.BLACK = '#000000'
        self.DARK_SQUARE_COLOR = '#8B4513'
        self.LIGHT_SQUARE_COLOR = '#DEB887'

        # Fonts
        self.universal_font = tkFont.Font(family="Helvetica", size=20)
        self.header_font = tkFont.Font(family="Helvetica", size=20, weight="bold")
        self.player_font = tkFont.Font(family="Helvetica", size=30, weight="bold")

        # Images
        self.piece_photos = {}
        for piece in 'pnbrqk':
            self.piece_photos[piece] = tk.PhotoImage(file='./res/black_%s.png' % piece)
        for piece in 'PNBRQK':
            self.piece_photos[piece] = tk.PhotoImage(file='./res/white_%s.png' % piece)

        # Stats Layout
        self.stats_bar = tk.Frame(self.win, width=frame_width, height=self.square_size, bg=self.WHITE)
        self.stats_bar.pack(fill='both', expand=1)

        white_player = "Player 1"
        black_player = "Player 2"
        if names is not None:
            white_player = names[0]
            black_player = names[1]

        w = tk.Frame(self.stats_bar, bg=self.WHITE, width=int((frame_width-self.half_square)/2), height=self.square_size)
        w.pack_propagate(0)
        w.pack(side=tk.LEFT)
        b = tk.Frame(self.stats_bar, bg=self.BLACK, width=int((frame_width-self.half_square)/2), height=self.square_size)
        b.pack_propagate(0)
        b.pack(side=tk.RIGHT)
        # white_king = tk.Label(w, image=self.piece_photos['K'], bg=self.WHITE)
        white_label = tk.Label(w, text=white_player, font=self.player_font, bg=self.WHITE)
        black_label = tk.Label(b, text=black_player, font=self.player_font, bg=self.BLACK, fg=self.WHITE)
        # black_king = tk.Label(b, image=self.piece_photos['K'], bg=self.BLACK)

        # white_king.grid(row=0, column=0)
        # white_label.grid(row=0, column=0)
        # black_label.grid(row=0, column=0)
        # black_king.grid(row=0, column=1)

        white_label.pack(fill='both', expand=1)
        black_label.pack(fill='both', expand=1)

        # Game Layout
        self.gameFrame = tk.Frame(self.win, width=frame_width, height=frame_height, bg=self.WHITE)
        self.gameFrame.pack(fill='both', expand=1)

        # --- headers ---
        self.empty = tk.Frame(self.gameFrame, width=self.half_square, height=int(self.square_size/2), bg=self.WHITE)
        self.row_headers = tk.Frame(self.gameFrame, width=self.half_square, height=self.square_size * 8, bg=self.WHITE)
        self.column_headers = tk.Frame(self.gameFrame, width=self.square_size * 8, height=self.half_square,
                                       bg=self.WHITE)

        self.empty.grid(row=0, column=0)
        self.row_headers.grid(row=1, column=0)
        self.column_headers.grid(row=0, column=1)

        j = 0
        num_frames = []
        for i in range(8, 0, -1):
            f = tk.Frame(self.row_headers, height=self.square_size, width=self.half_square, bg=self.WHITE)
            f.pack_propagate(0)
            f.grid(row=j, column=0)
            num_frame = tk.Label(f, text=i, anchor='center', font=self.header_font, bg=self.WHITE)
            num_frame.pack(fill='both', expand=1)
            num_frames.append(num_frame)
            j += 1

        j = 0
        let_frames = []
        for i in 'ABCDEFGH':
            f = tk.Frame(self.column_headers, height=self.half_square, width=self.square_size, bg=self.WHITE)
            f.pack_propagate(0)
            f.grid(row=0, column=j)
            let_frame = tk.Label(f, text=i, anchor='s', font=self.header_font, bg=self.WHITE)
            let_frame.pack(fill='both', expand=1)
            let_frames.append(let_frame)
            j += 1

        # --- chessboard ---
        self.canvas = tk.Canvas(self.gameFrame, width=self.square_size * 8, height=self.square_size * 8)

        for ridx, rname in enumerate(list('87654321')):
            for fidx, fname in enumerate(list('abcdefgh')):
                tag = str(ridx * fidx + fidx)
                color = [self.LIGHT_SQUARE_COLOR, self.DARK_SQUARE_COLOR][(ridx - fidx) % 2]
                # tag = ['light', 'dark'][(ridx - fidx) % 2]

                self.canvas.create_rectangle(
                    fidx * self.square_size, ridx * self.square_size,
                    fidx * self.square_size + self.square_size, ridx * self.square_size + self.square_size,
                    outline=color, fill=color, tag=tag)

        self.canvas.grid(row=1, column=1)

    def update_board(self, fen):
        # Change fen to 64-char string
        string_board = ''
        for f in fen:
            if f.isalpha():
                string_board += f
            elif f.isnumeric():
                for i in range(int(f)):
                    string_board += '0'

        for num, p in enumerate(list(string_board)):
            item = self.canvas.find_withtag(str(num + 1))
            coords = self.canvas.coords(item)

            # First, clear the chessboard
            ridx = num % 8
            fidx = int(num / 8)
            tag = str(ridx * fidx + fidx)
            color = [self.LIGHT_SQUARE_COLOR, self.DARK_SQUARE_COLOR][(ridx - fidx) % 2]

            self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], outline=color, fill=color,
                                         tag=tag)

            # If p is a piece, place the image of that piece
            if p in 'pnbrqkPNBRQK':
                # item = self.canvas.find_withtag(num+1)
                # coords = self.canvas.coords(item)
                image = self.piece_photos[p]
                offset_x = (self.square_size - image.width()) / 2
                offset_y = (self.square_size - image.height()) / 2

                self.canvas.create_image(coords[0] + offset_x, coords[1] + offset_y, image=image, state=tk.NORMAL,
                                         anchor=tk.NW, tag='piece')

        self.win.update()

    def game_over(self, message):
        top = tk.Toplevel(self.win, bg=self.WHITE)
        top.title("Game Over")

        w = 300
        h = 200
        ws = self.win.winfo_screenwidth()  # width of the screen
        hs = self.win.winfo_screenheight()  # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        top.geometry('%dx%d+%d+%d' % (w, h, x, y))

        msg1 = tk.Message(top, width=w - 2, text=message, font=self.universal_font, bg=self.WHITE)
        msg1.pack(fill=tk.BOTH, expand=1)

        button = tk.Button(top, text="Exit", font=self.universal_font, command=top.destroy, bg=self.WHITE)
        button.pack()

        button.wait_window(top)

if __name__ == '__main__':
    recon_game = ChessboardGUI()
    recon_game.update_board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')
    recon_game.game_over("Done!")
    recon_game.win.mainloop()