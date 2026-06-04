"""Finite fields used by the VDAF proof-of-concept code.

This vendored copy is Sage-free. Field elements are represented as Python
integers reduced modulo the class modulus, and polynomial interpolation uses a
direct Lagrange implementation over those field elements.
"""

import secrets

from typing_extensions import Self, TypeVar

from vdaf_poc.common import from_le_bytes, to_le_bytes


class Field:
    """The base class for finite fields."""

    MODULUS: int
    ENCODED_SIZE: int

    def __init__(self, val: int):
        value = int(val)
        assert value < self.MODULUS
        self.val = value % self.MODULUS

    @classmethod
    def zeros(cls, length: int) -> list[Self]:
        return [cls(0) for _ in range(length)]

    @classmethod
    def rand_vec(cls, length: int) -> list[Self]:
        """Return a random vector of field elements of length ``length``."""
        return [cls(secrets.randbelow(cls.MODULUS)) for _ in range(length)]

    @classmethod
    def encode_vec(cls, vec: list[Self]) -> bytes:
        """Encode a vector of field elements as a byte string."""
        encoded = bytes()
        for x in vec:
            encoded += to_le_bytes(x.as_unsigned(), cls.ENCODED_SIZE)
        return encoded

    @classmethod
    def decode_vec(cls, encoded: bytes) -> list[Self]:
        """Parse a vector of field elements from ``encoded``."""
        L = cls.ENCODED_SIZE
        if len(encoded) % L != 0:
            raise ValueError(
                "input length must be a multiple of the size of an "
                "encoded field element"
            )

        vec = []
        for i in range(0, len(encoded), L):
            x = from_le_bytes(encoded[i:i + L])
            if x >= cls.MODULUS:
                raise ValueError("modulus overflow")
            vec.append(cls(x))
        return vec

    @classmethod
    def encode_into_bit_vector(cls, val: int, bits: int) -> list[Self]:
        """Encode ``val`` as ``bits`` little-endian field bits."""
        if val >= 2 ** bits:
            raise ValueError(
                "Number of bits is not enough to represent the input integer."
            )
        return [cls((val >> bit) & 1) for bit in range(bits)]

    @classmethod
    def decode_from_bit_vector(cls, vec: list[Self]) -> Self:
        """Decode a little-endian bit vector into a field element."""
        bits = len(vec)
        if cls.MODULUS >> bits == 0:
            raise ValueError(
                "Number of bits is too large to be represented by field modulus."
            )
        decoded = cls(0)
        for bit_index, bit in enumerate(vec):
            decoded += cls(1 << bit_index) * bit
        return decoded

    def __add__(self, other: Self) -> Self:
        return self.__class__((self.val + other.val) % self.MODULUS)

    def __neg__(self) -> Self:
        return self.__class__((-self.val) % self.MODULUS)

    def __mul__(self, other: Self) -> Self:
        return self.__class__((self.val * other.val) % self.MODULUS)

    def inv(self) -> Self:
        return self.__class__(pow(self.val, -1, self.MODULUS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Field):
            return NotImplemented
        return self.__class__ is other.__class__ and self.val == other.val

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __div__(self, other: Self) -> Self:
        return self * other.inv()

    def __truediv__(self, other: Self) -> Self:
        return self * other.inv()

    def __pow__(self, n: int) -> Self:
        return self.__class__(pow(self.val, n, self.MODULUS))

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return str(self.val)

    def __hash__(self) -> int:
        return hash((self.__class__, self.val))

    def as_unsigned(self) -> int:
        return self.val


class NttField(Field):
    """A field with a power-of-two multiplicative subgroup."""

    GEN_ORDER: int

    @classmethod
    def gen(cls) -> Self:
        raise NotImplementedError()


class Field2(Field):
    """The finite field GF(2)."""

    MODULUS = 2
    ENCODED_SIZE = 1

    def conditional_select(self, inp: bytes) -> bytes:
        """Return ``inp`` if this element is one, otherwise zero bytes."""
        m = 0
        v = self.as_unsigned()
        for i in range(8):
            m |= v << i
        return bytes(map(lambda x: m & x, inp))


class Field64(NttField):
    """The finite field GF(2^32 * 4294967295 + 1)."""

    MODULUS = 2**32 * 4294967295 + 1
    GEN_ORDER = 2**32
    ENCODED_SIZE = 8

    @classmethod
    def gen(cls) -> Self:
        return cls(7) ** 4294967295


class Field96(NttField):
    """The finite field GF(2^64 * 4294966555 + 1)."""

    MODULUS = 2**64 * 4294966555 + 1
    GEN_ORDER = 2**64
    ENCODED_SIZE = 12

    @classmethod
    def gen(cls) -> Self:
        return cls(3) ** 4294966555


class Field128(NttField):
    """The finite field GF(2^66 * 4611686018427387897 + 1)."""

    MODULUS = 2**66 * 4611686018427387897 + 1
    GEN_ORDER = 2**66
    ENCODED_SIZE = 16

    @classmethod
    def gen(cls) -> Self:
        return cls(7) ** 4611686018427387897


class Field255(Field):
    """The finite field GF(2^255 - 19)."""

    MODULUS = 2**255 - 19
    ENCODED_SIZE = 32


F = TypeVar("F", bound=Field)


def poly_strip(field: type[F], p: list[F]) -> list[F]:
    """Remove leading zero coefficients from ``p``."""
    for i in reversed(range(len(p))):
        if p[i] != field(0):
            return p[:i + 1]
    return []


def poly_add(field: type[F], p: list[F], q: list[F]) -> list[F]:
    """Add two polynomials."""
    length = max(len(p), len(q))
    r = field.zeros(length)
    for i in range(length):
        if i < len(p):
            r[i] += p[i]
        if i < len(q):
            r[i] += q[i]
    return poly_strip(field, r)


def poly_mul(field: type[F], p: list[F], q: list[F]) -> list[F]:
    """Multiply two polynomials."""
    if len(p) == 0 or len(q) == 0:
        return []
    r = field.zeros(len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return poly_strip(field, r)


def poly_eval(field: type[F], p: list[F], eval_at: F) -> F:
    """Evaluate a polynomial at a point."""
    p = poly_strip(field, p)
    if len(p) == 0:
        return field(0)

    result = p[-1]
    for c in reversed(p[:-1]):
        result *= eval_at
        result += c
    return result


def poly_interp(field: type[F], xs: list[F], ys: list[F]) -> list[F]:
    """Compute the Lagrange interpolation polynomial for the given points."""
    if len(xs) != len(ys):
        raise ValueError("point/value length mismatch")
    if len(xs) == 0:
        return []

    result: list[F] = []
    for i, x_i in enumerate(xs):
        basis = [field(1)]
        denom = field(1)
        for j, x_j in enumerate(xs):
            if i == j:
                continue
            basis = poly_mul(field, basis, [-x_j, field(1)])
            denom *= x_i - x_j
        scale = ys[i] * denom.inv()
        term = [scale * coeff for coeff in basis]
        result = poly_add(field, result, term)
    return poly_strip(field, result)
