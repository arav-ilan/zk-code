"""Sage-free field definitions for the batched DZK package."""

from typing_extensions import Self

from vdaf_poc.field import Field64, Field96, Field128, NttField


class Field32(NttField):
    """The finite field GF(4293918721)."""

    MODULUS = 4293918721
    GEN_ORDER = 2**20
    ENCODED_SIZE = 4

    @classmethod
    def gen(cls) -> Self:
        return cls(3925978153)


class Field40(NttField):
    """The finite field GF(2^20 * 1048555 + 1)."""

    MODULUS = 2**20 * 1048555 + 1
    GEN_ORDER = 2**20
    ENCODED_SIZE = 5

    @classmethod
    def gen(cls) -> Self:
        return cls(7) ** 1048555


class Field48(NttField):
    """The finite field GF(2^40 * 205 + 1)."""

    MODULUS = 2**40 * 205 + 1
    GEN_ORDER = 2**40
    ENCODED_SIZE = 6

    @classmethod
    def gen(cls) -> Self:
        return cls(187322177118125)


class Field56(NttField):
    """The finite field GF(2^51 * 17 + 1)."""

    MODULUS = 2**51 * 17 + 1
    GEN_ORDER = 2**51
    ENCODED_SIZE = 7

    @classmethod
    def gen(cls) -> Self:
        return cls(129140163)


class Field72(NttField):
    """The finite field GF(2^66 * 7^2 + 1)."""

    MODULUS = 2**66 * 7**2 + 1
    GEN_ORDER = 2**66
    ENCODED_SIZE = 9

    @classmethod
    def gen(cls) -> Self:
        return cls(3) ** 49


class Field80(NttField):
    """The finite field GF(2^75 * 29 + 1)."""

    MODULUS = 2**75 * 29 + 1
    GEN_ORDER = 2**75
    ENCODED_SIZE = 10

    @classmethod
    def gen(cls) -> Self:
        return cls(3) ** 29


class Field88(NttField):
    """The finite field GF(2^72 * 60525 + 1)."""

    MODULUS = 2**72 * 60525 + 1
    GEN_ORDER = 2**72
    ENCODED_SIZE = 11

    @classmethod
    def gen(cls) -> Self:
        return cls(11) ** 60525


class Field104(NttField):
    """The finite field GF(2^88 * 49351 + 1)."""

    MODULUS = 2**88 * 49351 + 1
    GEN_ORDER = 2**53
    ENCODED_SIZE = 13

    @classmethod
    def gen(cls) -> Self:
        return cls(3) ** 49351


class Field112(NttField):
    """The finite field GF(2^96 * 58957 + 1)."""

    MODULUS = 2**96 * 58957 + 1
    GEN_ORDER = 2**96
    ENCODED_SIZE = 14

    @classmethod
    def gen(cls) -> Self:
        return cls(5) ** 58957


class Field120(NttField):
    """The finite field GF(2^104 * 41743 + 1)."""

    MODULUS = 2**104 * 41743 + 1
    GEN_ORDER = 2**104
    ENCODED_SIZE = 15

    @classmethod
    def gen(cls) -> Self:
        return cls(3) ** 41743


__all__ = [
    "NttField",
    "Field32",
    "Field40",
    "Field48",
    "Field56",
    "Field64",
    "Field72",
    "Field80",
    "Field88",
    "Field96",
    "Field104",
    "Field112",
    "Field120",
    "Field128",
]
