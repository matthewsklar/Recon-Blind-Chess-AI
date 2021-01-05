#!/usr/bin/env python3

"""
File Name:      player.py
Authors:        Michael Johnson and Leng Ghuy
Date:           03/13/2019

Description:    Python file for player agents containing common methods.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import os
import sys
import importlib
import inspect
import chess


class Player(object):
    def __init__(self):
        pass


def load_player(source_path):
    """
    This is function loads a subclass of the Player class that is contained in a python source file or python module.
    There must only be *1* such subclass in the file or module.

    :param source_path: the path to the source file to load
    :return: Tuple where the first element is the name of the loaded class, and the second element is the class type
    """

    if os.path.exists(source_path):
        # get the path to the main source file
        abs_source_path = os.path.abspath(source_path)

        # insert the directory of the bot source file into system path so we can import it
        # note: insert it first so we know we are searching this first
        sys.path.insert(0, os.path.dirname(abs_source_path))

        # import_module expects a module name, so remove the extension
        module_name = os.path.splitext(os.path.basename(abs_source_path))[0]
    else:
        module_name = source_path

    module = importlib.import_module(module_name)
    players = inspect.getmembers(module, lambda o: inspect.isclass(o) and issubclass(o, Player) and o != Player)
    if len(players) == 0:
        raise RuntimeError('{} did not contain any subclasses of {}'.format(source_path, Player))
    elif len(players) > 1:
        raise RuntimeError(
            '{} contained multiple subclasses of {}: {}. Should have exactly 1'.format(source_path, players, Player))
    return players[0]