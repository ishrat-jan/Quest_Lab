from pde_service.data.download import row_byte_range


def test_first_row_starts_at_tensor_offset():
    assert row_byte_range(0, 0, n_time=201, n_x=1024, offset=2048) == (2048, 2048 + 4096 - 1)


def test_row_offsets_follow_c_order_layout():
    a, b = row_byte_range(3, 7, n_time=201, n_x=1024, offset=2048)
    assert a == 2048 + (3 * 201 + 7) * 4096
    assert b - a + 1 == 4096


def test_last_row_ends_at_tensor_end():
    # full tensor (10000, 201, 1024) float32 ends at offset + 8,232,960,000 bytes
    _, b = row_byte_range(9999, 200, n_time=201, n_x=1024, offset=2048)
    assert b + 1 == 2048 + 10000 * 201 * 1024 * 4
