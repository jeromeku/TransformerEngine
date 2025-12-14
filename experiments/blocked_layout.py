import torch


torch.set_printoptions(
    threshold=100000000,  # print all data (without ... skipping) - can be huge!
    sci_mode=False,  # print all data on the same scale of 1 (this disables scientific notation)
    precision=0,  # print X decimal points for floats (default 4)
    edgeitems=5,  # when the data is large and skipped, control how many entries are printed on each edge
    linewidth=120,  # redefine linewidth for when lines are \n-wrapped in printout (default 80)
    # if threshold is defined, matrix printing will ignore this setting
    profile="full",  # printing defaults: "default", "short", "full"
)
BLOCK_M = 128
BLOCK_K = 4
BLOCK_SCALE_TILE_SHAPE = (BLOCK_M, BLOCK_K)


def print_coordinate_matrix(offsets: torch.Tensor, row_stride: int = BLOCK_K) -> None:
    """Pretty-print the row/column coordinate for every row-major offset."""

    if offsets.dim() != 2:
        raise ValueError("Expected a 2D tensor of row-major offsets")

    num_rows, num_cols = offsets.shape
    coordinate_strings = []
    max_len = 0

    for offset in offsets.reshape(-1):
        row = offset // row_stride
        col = offset % row_stride
        coord_str = f"({row}, {col})"
        coordinate_strings.append(coord_str)
        max_len = max(max_len, len(coord_str))

    max_len += 1  # add one space as padding between entries
    iterator = iter(coordinate_strings)
    for _ in range(num_rows):
        row_entries = [next(iterator) for _ in range(num_cols)]
        print("".join(entry.ljust(max_len) for entry in row_entries))


def _tile_to_shape(t: torch.Tensor, tile_shape: tuple = BLOCK_SCALE_TILE_SHAPE):
    """Reshape tensor to tile of shape tile_shape

    Args:
        t (torch.Tensor): _description_
        tile_shape (tuple, optional): _description_. Defaults to BLOCK_SCALE_TILE_SHAPE.
    Returns:
        3-D tensor of (num_tiles, *TILE_SHAPE)
    """

    assert t.ndim == 2

    for d, tile_d in zip(t.shape, tile_shape):
        assert d % tile_d == 0
   
    row_tile, col_tile = tile_shape
    rows, cols = t.shape
    num_row_tiles = rows // row_tile
    num_col_tiles = cols // col_tile
    t = t.reshape(num_row_tiles, row_tile, num_col_tiles, col_tile)
    t = t.permute(0, 2, 1, 3)  # (num_row_tiles, num_col_tiles, row_tile, col_tile)
    return t.reshape(-1, row_tile, col_tile)


def print_shape_and_stride(t: torch.Tensor):
    print(f"{tuple(t.shape)}:{tuple(t.stride())} {t.is_contiguous()=}")

def prof_and_print(f: callable, prof: torch.profiler.profile, prof_opts: dict):
    pass

def to_blocked(t: torch.Tensor):
    assert t.shape == torch.Size(BLOCK_SCALE_TILE_SHAPE)

    rearranged = t.reshape(4, 32, 4)  # shape: 4 x (32 x 4), strides: [128, 4, 1]
    tiled = rearranged.permute(1, 0, 2)  # (32 x 4) x 4, strides: [4, 128, 1]
    tile_32x16 = tiled.reshape(32, 16)  # (32, 16), strides: [128, 1]
    return tile_32x16
    
if __name__ == "__main__":
    NUM_TILES = 2
    TILE_NUMEL = NUM_TILES * BLOCK_K * NUM_TILES * BLOCK_M
    t = torch.arange(TILE_NUMEL, device="cuda:0").reshape(NUM_TILES * BLOCK_M, NUM_TILES * BLOCK_K)
    t2 = _tile_to_shape(t)
    print_shape_and_stride(t2)
    for tile in t2:
        blocked = to_blocked(tile)
        print_coordinate_matrix(blocked)
        print()

    if False:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        )
        print_opts = dict(sort_by="cuda_time_total", header="Rearranged")
        with prof:
            rearranged = t.reshape(4, 32, 4)  # shape: 4 x (32 x 4), strides: [128, 4, 1]
        print(prof.key_averages().table(**print_opts))
        print_shape_and_stride(rearranged)
        
        with prof:
            tiled = rearranged.permute(1, 0, 2)  # (32 x 4) x 4, strides: [4, 128, 1]
        print(prof.key_averages().table(**print_opts))
        print_shape_and_stride(tiled)

        with prof:
            tile_32x16 = tiled.reshape(32, 16)  # (32, 16), strides: [128, 1]
        print(prof.key_averages().table(max_name_column_width=100, **print_opts))
        print_shape_and_stride(tile_32x16)
