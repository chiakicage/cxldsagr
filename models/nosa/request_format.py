"""Request layout matching NOSA-8B's chat template with thinking disabled."""

from .infer import DEFAULT_MODEL_PATH

TOKENIZER_PATH = DEFAULT_MODEL_PATH / "tokenizer.json"
USER_LENGTHS = (4096, 16384)
ITEM_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
MAX_INPUT_TOKENS = 32768
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>")
ENDING = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
# SentencePiece prepends a dummy space to independently encoded blocks. Encode
# continuation blocks after a newline, matching their actual prompt boundary.
BLOCK_CONTEXT = "\n"
# Closing sentences cost 3, 5 or 6 tokens. Every remainder >= 8 is representable.
TAIL_RESERVE_TOKENS = 8


def prefix(instruction: str) -> str:
    return f"<|im_start|>user\n{instruction}\nUser history:\n"
