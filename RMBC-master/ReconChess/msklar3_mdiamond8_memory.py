class GameMemory:
    '''
    Stored game information

    turns := array of TurnMemory objects for each turn in the game
    v := actual outcome of the game
    game_in_play := whether this game is currently in play
    '''
    def __init__(self):
        self.turns = []
        self.v = None
        self.game_in_play = False

    def add_turn(self, turn):
        self.turns.append(turn)

    def to_string(self):
        string = ''
        
        string += '{value}\n'.format(value=self.v)
        
        for turn in self.turns:
            string += turn.to_string()

        return string

class TurnMemory:
    '''
    Stored turn information

    state := state of the game
    p := probability distribution from MCTS of taking actions at time step t
    '''
    def __init__(self, state, p):
        self.state = state  # state and pseudo-legal moves
        self.p = p

    def to_string(self):
        return '{state}, {p}\n'.format(
            state=str(self.state.detach()),
            p=str(self.p)
        )
