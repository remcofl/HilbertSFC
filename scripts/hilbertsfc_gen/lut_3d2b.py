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
NBITS_CHUNK = 2
BB_ENTRIES = 1 << (3 * NBITS_CHUNK)  # 6-bit bb (xxyyzz)
TABLE_SIZE = N_STATES * BB_ENTRIES


def generate_luts_3d2b() -> tuple[np.ndarray, np.ndarray]:
    """Generate 3D 2-bit flat LUTs.

    Returns
    -------
    (lut_3d2b_sb_so, lut_3d2b_so_sb)

    Both are uint16 arrays of length 24*64.

    Each entry packs a 6-bit symbol plus the next FSM state:
      packed = (next_state << 6) | symbol

    Encoding layout:
    - lut_3d2b_sb_so[(state << 6) | bb] = (next_state << 6) | oo
    - lut_3d2b_so_sb[(state << 6) | oo] = (next_state << 6) | bb

    Where:
    - bb is a 6-bit input symbol representing xxyyzz (two bits per axis)
    - oo is a 6-bit output symbol representing two 3D octants
    """

    lut_sb_so = np.zeros(TABLE_SIZE, dtype=np.uint16)
    lut_so_sb = np.zeros(TABLE_SIZE, dtype=np.uint16)

    for state in range(N_STATES):
        for b_packed in range(BB_ENTRIES):
            o_packed = 0
            s_next = state

            for bit in range(NBITS_CHUNK - 1, -1, -1):
                b_x = (b_packed >> (2 * NBITS_CHUNK + bit)) & 0x1
                b_y = (b_packed >> (1 * NBITS_CHUNK + bit)) & 0x1
                b_z = (b_packed >> (0 * NBITS_CHUNK + bit)) & 0x1
                b = (b_x << 2) | (b_y << 1) | b_z

                sb = (s_next << 3) | b
                o = int(LUT3D_SB_TO_O[sb])
                o_packed = (o_packed << 3) | o
                s_next = int(LUT3D_SB_TO_NEXT[sb])

            packed_so = np.uint16((s_next << 6) | o_packed)
            lut_sb_so[(state << 6) | b_packed] = packed_so

            packed_sb = np.uint16((s_next << 6) | b_packed)
            lut_so_sb[(state << 6) | o_packed] = packed_sb

    return lut_sb_so, lut_so_sb
