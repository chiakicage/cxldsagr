"""Historical DeepSeek V3.2 request layout and long-context experiment budgets."""

from pathlib import Path

TOKENIZER_PATH = Path("/mnt/nfs/share/models/DeepSeek-V3.2/tokenizer.json")
USER_LENGTHS = (4096, 16384, 65536, 262144, 1048576)
ITEM_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096)
MAX_INPUT_TOKENS = 1048576 + 4096
SPECIAL_TOKENS = ("<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>")
ENDING = "<｜Assistant｜></think>"
BLOCK_CONTEXT = ""
TAIL_RESERVE_TOKENS = 2


def prefix(instruction: str) -> str:
    return f"<｜begin▁of▁sentence｜>{instruction}<｜User｜>User history:\n"
