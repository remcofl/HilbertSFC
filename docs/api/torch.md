# `hilbertsfc.torch`

### Core APIs

#### Hilbert
- [`hilbert_encode_2d`][hilbertsfc.torch.hilbert_encode_2d]
- [`hilbert_decode_2d`][hilbertsfc.torch.hilbert_decode_2d]
- [`hilbert_encode_3d`][hilbertsfc.torch.hilbert_encode_3d]
- [`hilbert_decode_3d`][hilbertsfc.torch.hilbert_decode_3d]

#### Morton
- [`morton_encode_2d`][hilbertsfc.torch.morton_encode_2d]
- [`morton_decode_2d`][hilbertsfc.torch.morton_decode_2d]
- [`morton_encode_3d`][hilbertsfc.torch.morton_encode_3d]
- [`morton_decode_3d`][hilbertsfc.torch.morton_decode_3d]

### Cache Management

- [`precache_compile_luts`][hilbertsfc.torch.precache_compile_luts]
- [`clear_torch_lut_caches`][hilbertsfc.torch.clear_torch_lut_caches]

::: hilbertsfc.torch
    options:
      group_by_category: false
      members:
        - hilbert_encode_2d
        - hilbert_decode_2d
        - hilbert_encode_3d
        - hilbert_decode_3d
        - morton_encode_2d
        - morton_decode_2d
        - morton_encode_3d
        - morton_decode_3d
        - precache_compile_luts
        - clear_torch_lut_caches
