DATA_SET = "QQP"
DATA_PATH = "/home/ganxin/fa/ais/workspace/data/questionBData/%s/%s_train.tsv"%(DATA_SET, DATA_SET)
VAL_SET_SIZE = 0
MODEL = "gemma"

MAX_LENGTH_Q = 374 - 1
MAX_LENGTH_A = 10 - 1
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2 # 384

MODEL_PATH = "Lucachen/gemma2b"