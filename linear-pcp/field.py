from typing_extensions import Self, cast
from common import from_le_bytes, to_le_bytes
from sage.rings.finite_rings.finite_field_constructor import FiniteFieldFactory
from sage.all import GF

class Field:
    """The base class for finite fields."""

    # The prime modulus that defines arithmetic in the field.
    MODULUS: int

    # Number of bytes used to encode each field element.
    ENCODED_SIZE: int

    gf: FiniteFieldFactory

    def __init__(self, val: int):
        assert int(val) < self.MODULUS
        self.val = self.gf(val)

    @classmethod
    def zeros(cls, length: int) -> list[Self]:
        vec = [cls(cls.gf.zero()) for _ in range(length)]
        return vec

    @classmethod
    def rand_vec(cls, length: int) -> list[Self]:
        """
        Return a random vector of field elements of length `length`.
        """
        vec = [cls(cls.gf.random_element()) for _ in range(length)]
        return vec

    # NOTE: The encode_vec() and decode_vec() methods are excerpted in
    # the document, de-indented, as the figure {{field-derived-methods}}.
    # Their width should be limited to 69 columns after de-indenting, or
    # 73 columns before de-indenting, to avoid warnings from xml2rfc.
    # ===================================================================
    @classmethod
    def encode_vec(cls, vec: list[Self]) -> bytes:
        """
        Encode a vector of field elements `vec` as a byte string.
        """
        encoded = bytes()
        for x in vec:
            encoded += to_le_bytes(x.as_unsigned(), cls.ENCODED_SIZE)
        return encoded

    @classmethod
    def decode_vec(cls, encoded: bytes) -> list[Self]:
        """
        Parse a vector of field elements from `encoded`.
        """
        L = cls.ENCODED_SIZE
        if len(encoded) % L != 0:
            raise ValueError(
                'input length must be a multiple of the size of an '
                'encoded field element')

        vec = []
        for i in range(0, len(encoded), L):
            encoded_x = encoded[i:i+L]
            x = from_le_bytes(encoded_x)
            if x >= cls.MODULUS:
                raise ValueError('modulus overflow')
            vec.append(cls(x))
        return vec

    # NOTE: The encode_into_bit_vector() and decode_from_bit_vector()
    # methods are excerpted in the document, de-indented, as the figure
    # {{field-bit-rep}}. Their width should be limited to 69 columns
    # after de-indenting, or 73 columns before de-indenting, to avoid
    # warnings from xml2rfc.
    # ===================================================================
    @classmethod
    def encode_into_bit_vector(
            cls,
            val: int,
            bits: int) -> list[Self]:
        """
        Encode the bit representation of `val` with at most `bits` number
        of bits, as a vector of field elements.

        Pre-conditions:

            - `val >= 0`
            - `bits >= 0`
        """
        if val >= 2 ** bits:
            # Sanity check we are able to represent `val` with `bits`
            # number of bits.
            raise ValueError("Number of bits is not enough to represent "
                             "the input integer.")
        encoded = []
        for l in range(bits):
            encoded.append(cls((val >> l) & 1))
        return encoded

    @classmethod
    def decode_from_bit_vector(cls, vec: list[Self]) -> Self:
        """
        Decode the field element from the bit representation, expressed
        as a vector of field elements `vec`.
        """
        bits = len(vec)
        if cls.MODULUS >> bits == 0:
            raise ValueError("Number of bits is too large to be "
                             "represented by field modulus.")
        decoded = cls(0)
        for (l, bit) in enumerate(vec):
            decoded += cls(1 << l) * bit
        return decoded

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.val + other.val)

    def __neg__(self) -> Self:
        return self.__class__(-self.val)

    def __mul__(self, other: Self) -> Self:
        return self.__class__(self.val * other.val)

    def inv(self) -> Self:
        return self.__class__(self.val**-1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Field):
            return NotImplemented
        return cast(bool, self.val == other.val)

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __div__(self, other: Self) -> Self:
        return self * other.inv()

    def __pow__(self, n: int) -> Self:
        return self.__class__(self.val ** n)

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return str(self.val)

    def as_unsigned(self) -> int:
        return int(self.gf(self.val))

class NttField(Field):
    """
    A field that is suitable for use with the NTT ("number theoretic
    transform") algorithm for efficient polynomial interpolation. Such a field
    defines a large multiplicative subgroup whose order is a power of 2.
    """

    # Order of the multiplicative group generated by `Field.gen()`.
    GEN_ORDER: int

    @classmethod
    def gen(cls) -> Self:
        raise NotImplementedError()

class Field32(NttField):
    """The finite field GF(4293918721)."""

    MODULUS = 4293918721
    GEN_ORDER = 2**20
    ENCODED_SIZE = 4

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls):
        return cls(3925978153)


class Field40(NttField):
    """
    The finite field GF(2^20 * 1048555 + 1).
    Modulus is selected using https://github.com/cfrg/draft-irtf-cfrg-vdaf/blob/main/misc/prime-hunt.md
    first one by running `search(20, 40, 200)`
    q = 5 * 43 * 4877 = 1048555
    p = 2^20 * q + 1
    generator = pow(7, q, p)
    """

    MODULUS = 2**20 * 1048555 + 1
    GEN_ORDER = 2**20
    ENCODED_SIZE = 5

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7) ** 1048555
    
class Field64(NttField):
    """The finite field GF(2^32 * 4294967295 + 1)."""

    MODULUS = 2**32 * 4294967295 + 1
    GEN_ORDER = 2**32
    ENCODED_SIZE = 8

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7)**4294967295


class Field96(NttField):
    """The finite field GF(2^64 * 4294966555 + 1)."""

    MODULUS = 2**64 * 4294966555 + 1
    GEN_ORDER = 2**64
    ENCODED_SIZE = 12

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(3)**4294966555

class Field128(NttField):
    """The finite field GF(2^66 * 4611686018427387897 + 1)."""

    MODULUS = 2**66 * 4611686018427387897 + 1
    GEN_ORDER = 2**66
    ENCODED_SIZE = 16

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7)**4611686018427387897


class Field255(Field):
    """The finite field GF(2^255 - 19)."""

    MODULUS = 2**255 - 19
    ENCODED_SIZE = 32

    # Sage finite field object.
    gf = GF(MODULUS)