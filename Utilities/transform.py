import numpy as np 
import torch
import Utilities.Parameters as Parameters



def _encode_list_state(dest_np, state, who_move):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
    """


    for row    in range(Parameters.GAMEROWS):
        for col in range(Parameters.GAMECOLS):

            if state[row][col] == 0:
                continue

            if state[row][col] == who_move:
                dest_np[0, row, col] = 1.0

            else:
                dest_np[1,row,col] = 1.0




def state_lists_to_batch(state_lists, who_moves_lists, device="cuda"):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
    #assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + Parameters.OBS_SHAPE, dtype=np.float32)
    for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
        _encode_list_state(batch[idx], state.BoardState , who_move)

    return torch.tensor(batch).to(device)


