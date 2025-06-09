import json
from core import TangoBoard
from game import recursive_tango


def main():
    board = TangoBoard()
    _, sols, _ = recursive_tango(position=36, tango_board=board, solutions=list())
    # Export only first 5 solutions to keep file small
    data = [sol.board.tolist() for sol in sols[:5]]
    with open('docs/solutions.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
