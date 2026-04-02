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


def _simulate_hilbert_traversal(
    start_state: int, b_packed: int, nbits: int
) -> tuple[int, int]:
    """Simulate nbits 1-bit Hilbert FSM iterations (MSB first).

    Parameters
    ----------
    start_state:
        Starting FSM state (0..3).
    b_packed:
        Packed input bits `b` with layout `(x_n..x_0, y_n..y_0)` (2*nbits bits).
    nbits:
        Number of 1-bit iterations.

    Returns
    -------
    (q_packed, next_state)
        - q_packed: packed quadrants (2*nbits bits)
        - next_state: FSM state after `nbits` iterations
    """

    s_next = start_state
    q_packed = 0

    for bit in reversed(range(nbits)):
        x = (b_packed >> (nbits + bit)) & 1
        y = (b_packed >> bit) & 1
        b = (x << 1) | y
        sb = (s_next << 2) | b
        q = int(LUT_SB_TO_Q[sb])
        s_next = int(LUT_SB_TO_NEXT[sb])
        q_packed = (q_packed << 2) | q

    return q_packed, s_next


def generate_luts_2dnb_compacted(tile_nbits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate compacted 2D tile-nbit Hilbert lookup tables (LUTs).

    This produces two 1D LUTs for a Hilbert tile of size `tile_nbits` bits per axis:

    - `lut_b_qs[b]`: maps packed input bits `b = (x_n << n) | y_n` to packed quadrants and next FSM state.
    - `lut_q_bs[q]`: maps packed quadrants `q` to packed input bits and next FSM state.

    Each table entry is a `uint64` containing 4 lanes (one per FSM state), each 16 bits:

        +-------------------------------+
        | lane3 | lane2 | lane1 | lane0 |
        +-------------------------------+

    Within each 16-bit lane:
    - The **high bits** store the packed quadrants bits `q`.
    - The **low bits** store the next FSM state bits `s`.
    - For `tile_nbits <= 5`, the next state is **pre-shifted by 4** (`state_next << 4`)
      so lanes can be directly selected via `lane = lut[b] >> state`.
    - For `tile_nbits >= 6`, the next state occupies the **lowest 2 bits**, and lanes
      require an extra shift during runtime to extract the next lane: `lane = lut[b] >> (state << 4)`.

    Lane layout varies with `tile_nbits` (example layouts):

        qqqqqqqq|qqqqqqss   (n=7)
        qqqqqqqq|00ss0000   (n=4)

    Parameters
    ----------
    tile_nbits : int
        Number of 1-bit FSM iterations per table lookup (bits per axis per step). Supported range: 1..7.

    Returns
    -------
    tuple of np.ndarray
        - lut_b_qs: uint64 array of length 2^(2*tile_nbits), mapping b -> packed (q, next_state)
        - lut_q_bs: uint64 array of length 2^(2*tile_nbits), mapping q -> packed (b, next_state)
    """

    if tile_nbits < 1 or tile_nbits > 7:
        raise ValueError("tile_nbits must be in [1, 7]")

    def _pack_lane_qs_16(q: int, s_next: int, n: int) -> int:
        q_shift = 16 - 2 * n
        s_payload = s_next & 0x3
        if n <= 5:
            s_payload = s_payload << 4

        return (q << q_shift) | s_payload

    entries = 1 << (2 * tile_nbits)
    lut_b_qs = np.zeros(entries, dtype=np.uint64)
    lut_q_bs = np.zeros(entries, dtype=np.uint64)

    for i in range(entries):
        for state in range(N_STATES):
            b_packed = i

            q_packed, s_next = _simulate_hilbert_traversal(state, b_packed, tile_nbits)

            lane_bits = _pack_lane_qs_16(q_packed, s_next, tile_nbits)
            s_lsh4 = state << 4

            # Pack encoding (b -> q)
            lut_b_qs[b_packed] |= np.uint64(lane_bits) << s_lsh4
            # Pack decoding (q -> b)
            # Reuse the same lane packing but with the b symbol in the q position.
            lane_bits_dec = _pack_lane_qs_16(b_packed, s_next, tile_nbits)
            lut_q_bs[q_packed] |= np.uint64(lane_bits_dec) << s_lsh4

    return lut_b_qs, lut_q_bs


def generate_luts_2dnb_flat(tile_nbits: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D tile-nbit flat (state-dependent) Hilbert lookup tables.

    This produces two 1D LUTs for a Hilbert tile with `tile_nbits` bits per coordinate.
    Tables are indexed by `(state, symbol)` (non-compacted form).

    Each table is a `uint16` array of length `4 * 2^(2*tile_nbits)`.

    Layouts
    -------
    Each entry packs the next FSM state and output symbol:

    Encoding (b -> q):
        lut_sb_sq[(state << (2*n)) | b] = (next_state << (2*n)) | q

    Decoding (q -> b):
        lut_sq_sb[(state << (2*n)) | q] = (next_state << (2*n)) | b

    Symbol format
    -------------
    b : 2*tile_nbits-bit input symbol (b = xy; tile_nbits bits per coordinate)
    q : 2*tile_nbits-bit output symbol (tile_nbits quadrants)

    Parameters
    ----------
    tile_nbits : int
        Bits per coordinate per lookup step (1-7).

    Returns
    -------
    tuple of np.ndarray
        (lut_sb_sq_u16, lut_sq_sb_u16)
    """
    if tile_nbits < 1 or tile_nbits > 7:
        raise ValueError("tile_nbits must be in [1, 7]")

    sym_bits = 2 * tile_nbits
    sym_entries = 1 << sym_bits
    table_size = N_STATES * sym_entries

    lut_sb_sq = np.zeros(table_size, dtype=np.uint16)
    lut_sq_sb = np.zeros(table_size, dtype=np.uint16)

    for state in range(N_STATES):
        state_base = state << sym_bits
        for b_packed in range(sym_entries):
            q_packed, s_next = _simulate_hilbert_traversal(state, b_packed, tile_nbits)

            packed_sq = np.uint16((s_next << sym_bits) | q_packed)
            lut_sb_sq[state_base | b_packed] = packed_sq

            packed_sb = np.uint16((s_next << sym_bits) | b_packed)
            lut_sq_sb[state_base | q_packed] = packed_sb

    return lut_sb_sq, lut_sq_sb
