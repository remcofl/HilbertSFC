import numpy as np

# 3-bit FSM tables --- b = (x, y, z)
# fmt: off
LUT3D_SB_TO_O = np.array([    # (state, b) -> o
    0, 1, 3, 2, 7, 6, 4, 5,
    0, 1, 7, 6, 3, 2, 4, 5,
    0, 3, 1, 2, 7, 4, 6, 5,
    0, 7, 1, 6, 3, 4, 2, 5,
    0, 3, 7, 4, 1, 2, 6, 5,
    0, 7, 3, 4, 1, 6, 2, 5,
    2, 1, 3, 0, 5, 6, 4, 7,
    6, 1, 7, 0, 5, 2, 4, 3,
    2, 3, 1, 0, 5, 4, 6, 7,
    6, 7, 1, 0, 5, 4, 2, 3,
    4, 3, 7, 0, 5, 2, 6, 1,
    4, 7, 3, 0, 5, 6, 2, 1,
    2, 1, 5, 6, 3, 0, 4, 7,
    6, 1, 5, 2, 7, 0, 4, 3,
    2, 3, 5, 4, 1, 0, 6, 7,
    6, 7, 5, 4, 1, 0, 2, 3,
    4, 3, 5, 2, 7, 0, 6, 1,
    4, 7, 5, 6, 3, 0, 2, 1,
    2, 5, 1, 6, 3, 4, 0, 7,
    6, 5, 1, 2, 7, 4, 0, 3,
    2, 5, 3, 4, 1, 6, 0, 7,
    6, 5, 7, 4, 1, 2, 0, 3,
    4, 5, 3, 2, 7, 6, 0, 1,
    4, 5, 7, 6, 3, 2, 0, 1,
], dtype=np.uint8)

LUT3D_SB_TO_NEXT = np.array([    # (state, b) -> next_state
    5,  1, 13,  0, 13, 22,  5,  0,
    3,  0,  7, 23,  7,  1,  3,  1,
    4, 19,  3,  2, 19,  4, 16,  2,
    1,  9,  2, 17,  9,  1,  3,  3,
    2, 21, 21,  2,  5,  4, 10,  4,
    0, 15, 15,  0,  4, 11,  5,  5,
    6,  7, 12, 11,  6, 20, 11, 12,
   21,  6,  1,  9,  7,  7,  9,  1,
    8, 18,  9, 10,  8, 10, 14, 18,
   15,  3,  8,  7,  9,  7,  9,  3,
    8, 23, 23,  8, 10, 10,  4, 11,
    6, 17, 17,  6, 11,  5, 11, 10,
   12, 13, 12, 18,  6, 17, 17,  6,
   19, 12, 13, 13,  0, 15, 15,  0,
   14, 20, 14, 16, 15, 16,  8, 20,
    9,  5, 15, 13, 14, 13, 15,  5,
   14, 22, 16, 16, 22, 14,  2, 17,
   12, 11, 17,  3, 11, 12, 17, 16,
   18, 18, 19, 12,  8, 23, 23,  8,
   13, 19, 18, 19,  2, 21, 21,  2,
   20, 20, 14, 22, 21,  6, 22, 14,
    7, 21,  4, 19, 20, 21, 19,  4,
   20, 22, 16, 22, 16,  0, 20, 23,
   18, 23, 10,  1, 10, 23, 18, 22,
], dtype=np.uint8)
# fmt: on

N_STATES = 24
DEFAULT_TILE_NBITS = 2


def generate_luts_3dnb_flat(tile_nbits: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate flat 3D Hilbert LUTs for a configurable tile width.

    Each table has ``24 * 2 ** (3 * tile_nbits)`` entries indexed by
    ``(state << symbol_bits) | symbol``. Entries pack the next 5-bit state
    above the transformed symbol. Tile widths up to 3 fit in ``uint16``;
    tile width 4 requires ``uint32``.
    """

    if tile_nbits < 1 or tile_nbits > 4:
        raise ValueError("tile_nbits must be in [1, 4]")

    symbol_bits = 3 * tile_nbits
    symbol_entries = 1 << symbol_bits
    table_size = N_STATES * symbol_entries
    dtype = np.uint16 if (5 + symbol_bits) <= 16 else np.uint32
    lut_sb_so = np.zeros(table_size, dtype=dtype)
    lut_so_sb = np.zeros(table_size, dtype=dtype)

    for state in range(N_STATES):
        for b_packed in range(symbol_entries):
            o_packed = 0
            s_next = state

            for bit in range(tile_nbits - 1, -1, -1):
                b_x = (b_packed >> (2 * tile_nbits + bit)) & 0x1
                b_y = (b_packed >> (1 * tile_nbits + bit)) & 0x1
                b_z = (b_packed >> (0 * tile_nbits + bit)) & 0x1
                b = (b_x << 2) | (b_y << 1) | b_z

                sb = (s_next << 3) | b
                o = int(LUT3D_SB_TO_O[sb])
                o_packed = (o_packed << 3) | o
                s_next = int(LUT3D_SB_TO_NEXT[sb])

            packed_so = dtype((s_next << symbol_bits) | o_packed)
            lut_sb_so[(state << symbol_bits) | b_packed] = packed_so

            packed_sb = dtype((s_next << symbol_bits) | b_packed)
            lut_so_sb[(state << symbol_bits) | o_packed] = packed_sb

    return lut_sb_so, lut_so_sb
