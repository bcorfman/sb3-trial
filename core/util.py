from dataclasses import dataclass


@dataclass(eq=False)
class Coord:
    row: int
    col: int

    def __eq__(self, other):
        if isinstance(other, Coord):
            return self.row == other.row and self.col == other.col
        return self.row == other[0] and self.col == other[1]

    def __add__(self, other):
        if isinstance(other, Coord):
            return Coord(self.row + other.row, self.col + other.col)
        else:
            return Coord(self.row + other[0], self.col + other[1])

    def __radd__(self, other):
        if isinstance(other, Coord):
            return self.__add__(other)
        else:
            return Coord(self.row + other[0], self.col + other[1])

    def __hash__(self):
        return hash((self.row, self.col))

    def as_tuple(self):
        return self.row, self.col

    @classmethod
    def from_tuple(cls, t):
        coord = cls.__new__(cls)
        coord.row, coord.col = t
        return coord


def _manhattan_distance(loc1: Coord, loc2: Coord):
    return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
