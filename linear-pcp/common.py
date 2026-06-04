def to_le_bytes(val: int, length: int) -> bytes:
  """
  Convert unsigned integer `val` in range `[0, 2 ^ (8 * length))` to a
  little-endian byte string.
  """
  val = int(val)
  if val < 0 or val >= (1 << (8 * length)):
      raise ValueError(
          'bad to_le_bytes call: val=%d length=%d' % (val, length))
  return val.to_bytes(length, byteorder='little')


def from_le_bytes(encoded: bytes) -> int:
  """Parse an unsigned integer from a little-endian byte string."""
  return int.from_bytes(encoded, byteorder='little')
