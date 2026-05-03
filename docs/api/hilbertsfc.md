# `hilbertsfc`

### Core APIs

#### Hilbert

- [`hilbert_encode_2d`][hilbertsfc.hilbert_encode_2d]
- [`hilbert_decode_2d`][hilbertsfc.hilbert_decode_2d]
- [`hilbert_encode_3d`][hilbertsfc.hilbert_encode_3d]
- [`hilbert_decode_3d`][hilbertsfc.hilbert_decode_3d]

#### Morton

- [`morton_encode_2d`][hilbertsfc.morton_encode_2d]
- [`morton_decode_2d`][hilbertsfc.morton_decode_2d]
- [`morton_encode_3d`][hilbertsfc.morton_encode_3d]
- [`morton_decode_3d`][hilbertsfc.morton_decode_3d]

### Kernel Accessors

#### Hilbert

- [`get_hilbert_encode_2d_kernel`][hilbertsfc.get_hilbert_encode_2d_kernel]
- [`get_hilbert_decode_2d_kernel`][hilbertsfc.get_hilbert_decode_2d_kernel]
- [`get_hilbert_encode_3d_kernel`][hilbertsfc.get_hilbert_encode_3d_kernel]
- [`get_hilbert_decode_3d_kernel`][hilbertsfc.get_hilbert_decode_3d_kernel]

#### Morton

- [`get_morton_encode_2d_kernel`][hilbertsfc.get_morton_encode_2d_kernel]
- [`get_morton_decode_2d_kernel`][hilbertsfc.get_morton_decode_2d_kernel]
- [`get_morton_encode_3d_kernel`][hilbertsfc.get_morton_encode_3d_kernel]
- [`get_morton_decode_3d_kernel`][hilbertsfc.get_morton_decode_3d_kernel]

### Cache Management

- [`clear_all_caches`][hilbertsfc.clear_all_caches]
- [`clear_kernel_caches`][hilbertsfc.clear_kernel_caches]
- [`clear_lut_caches`][hilbertsfc.clear_lut_caches]

::: hilbertsfc
    options:
      members:
        - hilbert_encode_2d
        - hilbert_decode_2d
        - hilbert_encode_3d
        - hilbert_decode_3d
        - morton_encode_2d
        - morton_decode_2d
        - morton_encode_3d
        - morton_decode_3d
        - get_hilbert_encode_2d_kernel
        - get_hilbert_decode_2d_kernel
        - get_hilbert_encode_3d_kernel
        - get_hilbert_decode_3d_kernel
        - get_morton_encode_2d_kernel
        - get_morton_decode_2d_kernel
        - get_morton_encode_3d_kernel
        - get_morton_decode_3d_kernel
        - clear_lut_caches
        - clear_kernel_caches
        - clear_all_caches
