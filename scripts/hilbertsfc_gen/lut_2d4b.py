import numpy as np

# 2-bit FSM tables --- b = (x, y)
# fmt: off
LUT_SB_TO_Q = np.array(       # (state, b) -> q
    [0, 1, 3, 2,
     0, 3, 1, 2,
     2, 3, 1, 0,
     2, 1, 3, 0], dtype=np.uint8)

LUT_SB_TO_NEXT = np.array(    # (state, b) -> next_state
    [1, 0, 3, 0,
     0, 2, 1, 1,
     2, 1, 2, 3,
     3, 3, 0, 2], dtype=np.uint8)
# fmt: on

N_STATES = 4
NBITS_CHUNK = 4
B_ENTRIES = 1 << (2 * NBITS_CHUNK)  # 8-bit bbbb (xxxxyyyy)


def generate_luts_2d4b_compacted() -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D 4-bit (4 iterations) compacted Hilbert LUTs.

    This builds two 1D lookup tables, indexed by a single 8-bit value:

    - ``lut_2d4b_b_qs[bbbb]`` where ``bbbb = (xxxx << 4) | yyyy``.
    - ``lut_2d4b_q_bs[qqqq]`` where ``qqqq`` is 4 quadrants packed
      as 4 symbols of 2 bits each (total 8 bits).

    Each table entry is a ``uint64`` containing 4 independent 'lanes', one per
    FSM state, with each lane occupying 16 bits. The lane selection is done by
    shifting by ``state_shift = state << 4`` (i.e. 0, 16, 32, 48). This enables
    state-independent lookup (gather) into the table, with state selection
    performed afterward via a relatively cheap variable register shift rather
    than a state-dependent gather.

    Packed layout (lanes are states 0..3, low-to-high)::

                              uint64
        +-----------------------------------------------+
        | qqqq 0s00 | qqqq 0s00 | qqqq 0s00 | qqqq 0s00 |
        |     3     |     2     |     1     |     0     |  <- lane = state
        +-----------------------------------------------+

        qqqq: 8 bits, four 2-bit quadrants, MSB-first
        0s00: 8 bits, where s is the 2-bit next_state stored as (next_state<<4)

    For the decode table, the same pattern applies, except the upper byte is
    ``bbbb`` (the 8-bit packed input bits) instead of ``qqqq``.

    Returns
    -------
    (lut_2d4b_b_qs, lut_2d4b_q_bs)

    - lut_2d4b_b_qs: uint64 array of length 256 (bbbb -> qqqq, next_state)
    - lut_2d4b_q_bs: uint64 array of length 256 (qqqq -> bbbb, next_state)
    """
    lut_2d4b_b_qs = np.zeros(B_ENTRIES, dtype=np.uint64)
    lut_2d4b_q_bs = np.zeros(B_ENTRIES, dtype=np.uint64)

    for i in range(B_ENTRIES):
        for state in range(N_STATES):
            s_next = state
            qqqq = 0
            bbbb = i

            # Simulate 4 iterations of 1-bit FSM, MSB first
            for bit in reversed(range(NBITS_CHUNK)):
                x = (bbbb >> (4 + bit)) & 1
                y = (bbbb >> bit) & 1
                b = (x << 1) | y
                idx = (s_next << 2) | b
                q = int(LUT_SB_TO_Q[idx])
                s_next = int(LUT_SB_TO_NEXT[idx])
                qqqq = (qqqq << 2) | q

            s_lsh4 = state << 4
            s_next_shifted = (s_next << 4) << s_lsh4
            qqqq_shifted = qqqq << (s_lsh4 + 8)
            bbbb_shifted = bbbb << (s_lsh4 + 8)

            # Pack encoding
            lut_2d4b_b_qs[bbbb] |= qqqq_shifted | s_next_shifted
            # Pack decoding
            lut_2d4b_q_bs[qqqq] |= bbbb_shifted | s_next_shifted

    return lut_2d4b_b_qs, lut_2d4b_q_bs
