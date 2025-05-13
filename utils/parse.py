import os
import chess
import chess.pgn
import numpy as np
import torch
import zstandard as zstd
import io
from utils.fen_encoder import fen_to_tensor,FenEncoder
from utils.vocab import policy_index
from tqdm import tqdm

min_length = 20
num_moves = 1
block_size = 256
clip_length = block_size-64

def get_batch(pgn_path, return_fen = False, triple = False,batch_size = 16,all_elo = False,elo_threshold = 1650):
    if pgn_path.endswith(".zst"):
        file_compressed = open(pgn_path,'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(file_compressed)
        f = io.TextIOWrapper(reader, encoding='utf-8')
    else:
        f  = open(pgn_path, "r")
    fen_array = []
    moves_array = []

    while True:
        pgn = chess.pgn.read_game(f)
        if pgn.next()==None:
            continue
        elo = min(int(pgn.headers["WhiteElo"]),int(pgn.headers["BlackElo"]))
        if all_elo:
            elo = 3000
        if elo <= elo_threshold or 'FEN' in pgn.headers.keys() or '960' in pgn.headers['Event'] or 'Odds' in pgn.headers['Event'] or 'house' in pgn.headers['Event'] or 'Bullet' in pgn.headers['Event']:
            continue
        moves = [move for move in pgn.mainline_moves()]
        if len(moves) < min_length:
            continue
        #start index is a random int in 0, len(moves)- num_moves
        start_index = np.random.randint(0,len(moves)-num_moves)
        board = chess.Board()
        for move in moves[:start_index]:
            board.push(move)
        fen = board.fen()
        moves = moves[start_index:]
        fen_array.append(fen)
        moves_array.append(encode_moves(moves))
        if len(fen_array) == batch_size:
            if return_fen:
                yield (fen_array)
            elif triple:
                yield (encode_fens(fen_array).to("cuda"),clip_and_batch(moves_array).to("cuda"), fen_array)
            else:
                yield (encode_fens(fen_array).to("cuda"),clip_and_batch(moves_array).to("cuda"))
            fen_array = []
            moves_array = []

def dir_iterator(dir_path,return_fen = False,triple = False,batch_size = 16, all_elo = False):
    for pgn in os.listdir(dir_path):
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = get_batch(pgn_path,return_fen = return_fen, triple = triple,batch_size = batch_size, all_elo = all_elo)
        while True:
            try:
                yield next(gen)
            except:
                break
            
def encode_fens(fen_array):
    #encode in pytorch tensor
    #print(fen_array)
    fens = torch.from_numpy(np.array([fen_to_tensor(fen) for fen in fen_array]))
    return fens

def encode_moves(moves_array):
    moves = []
    #print(moves_array)
    for move in moves_array:
        if move.uci() in policy_index:
            move_id = policy_index.index(move.uci())
        else:
            move_id = policy_index.index(move.uci()[:-1])
        moves.append(move_id)
    return torch.from_numpy(np.array(moves))


def encode_moves_bis(moves_array):
    moves = []
    #print(moves_array)
    for move in moves_array:
        if move in policy_index:
            move_id = policy_index.index(move)
        else:
            move_id = policy_index.index(move[:-1])
        moves.append(move_id)
    return torch.from_numpy(np.array(moves))

def clip_and_batch(moves_array,clip = clip_length):
    #clip and batch moves
    moves = torch.full((len(moves_array),clip),1928,dtype = torch.int64)
    for i in range(len(moves_array)):
        if moves_array[i].shape[0] > clip:
            moves[i] = moves_array[i][:clip]
        else:
            moves[i,:moves_array[i].shape[0]] = moves_array[i]
    return moves

