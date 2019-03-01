#!/usr/bin/env python3
"""A sudoku solver that uses mixed-integer programming."""

# Rules of sudoku:
#
#  1. Each cell has exactly one number.
#  2. Each number can only be present in each row once.
#  3. Each number can only be present in each column once.
#  4. Each number can only be present in each 3x3 sub-grid once.
#
#   0 1 2   3 4 5   6 7 8
# 0       ┃       ┃       
# 1   0   ┃   1   ┃   2  
# 2       ┃       ┃      
#   ━━━━━━╈━━━━━━━╈━━━━━━
# 3       ┃       ┃      
# 4   3   ┃   4   ┃   5  
# 5       ┃       ┃      
#   ━━━━━━╈━━━━━━━╈━━━━━━
# 6       ┃       ┃      
# 7   6   ┃   7   ┃   8  
# 8       ┃       ┃      

from ortools.linear_solver import pywraplp

import argparse
import sys


_RULES = (
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)),  # row 0
    ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)),  # row 1
    ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)),  # row 2
    ((3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)),  # row 3
    ((4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)),  # row 4
    ((5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)),  # row 5
    ((6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8)),  # row 6
    ((7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)),  # row 7
    ((8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)),  # row 8

    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)),  # col 0
    ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)),  # col 1
    ((0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2)),  # col 2
    ((0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3)),  # col 3
    ((0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)),  # col 4
    ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5)),  # col 5
    ((0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6)),  # col 6
    ((0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)),  # col 7
    ((0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)),  # col 8

    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),  # box 0
    ((0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)),  # box 1
    ((0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8)),  # box 2
    ((3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)),  # box 3
    ((3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)),  # box 4
    ((3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8)),  # box 5
    ((6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2)),  # box 6 
    ((6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5)),  # box 7
    ((6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)),  # box 8
)


def prepare(solver, givens):
    board = {r: {c: {n: None for n in range(9)} for c in range(9)} for r in range(9)}
    for row in range(9):
        for col in range(9):
            # Create a boolean variable for each possible number in this cell.
            for num in range(9):
                board[row][col][num] = solver.BoolVar('%d.%d=%d' % (row, col, num))
            
            # Each cell must have exactly one number set.
            solver.Add(sum(board[row][col].values()) == 1)

            # Pin the cell to the given number, if necessary.
            if givens[row][col]:
                num = givens[row][col] - 1
                solver.Add(board[row][col][num] == 1)

    # Apply each rule to the board.
    for num in range(9):
        for rule in _RULES:
            solver.Add(sum(board[r][c][num] for (r, c) in rule) == 1)
    
    return board


def read_givens(path):
    givens = {r: {c: None for c in range(9)} for r in range(9)}
    with open(path) as f:
        for row, line in enumerate(f):
            for col, value in enumerate(line.split()):
                try:
                    givens[row][col] = int(value)
                except ValueError:
                    pass
    return givens


def print_solution(solution):
    print("""
{s[0][0]} {s[0][1]} {s[0][2]} | {s[0][3]} {s[0][4]} {s[0][5]} | {s[0][6]} {s[0][7]} {s[0][8]}
{s[1][0]} {s[1][1]} {s[1][2]} | {s[1][3]} {s[1][4]} {s[1][5]} | {s[1][6]} {s[1][7]} {s[1][8]}
{s[2][0]} {s[2][1]} {s[2][2]} | {s[2][3]} {s[2][4]} {s[2][5]} | {s[2][6]} {s[2][7]} {s[2][8]}
------+-------+------
{s[3][0]} {s[3][1]} {s[3][2]} | {s[3][3]} {s[3][4]} {s[3][5]} | {s[3][6]} {s[3][7]} {s[3][8]}
{s[4][0]} {s[4][1]} {s[4][2]} | {s[4][3]} {s[4][4]} {s[4][5]} | {s[4][6]} {s[4][7]} {s[4][8]}
{s[5][0]} {s[5][1]} {s[5][2]} | {s[5][3]} {s[5][4]} {s[5][5]} | {s[5][6]} {s[5][7]} {s[5][8]}
------+-------+------
{s[6][0]} {s[6][1]} {s[6][2]} | {s[6][3]} {s[6][4]} {s[6][5]} | {s[6][6]} {s[6][7]} {s[6][8]}
{s[7][0]} {s[7][1]} {s[7][2]} | {s[7][3]} {s[7][4]} {s[7][5]} | {s[7][6]} {s[7][7]} {s[7][8]}
{s[8][0]} {s[8][1]} {s[8][2]} | {s[8][3]} {s[8][4]} {s[8][5]} | {s[8][6]} {s[8][7]} {s[8][8]}
""".format(s=solution), end='')


def solve(solver, board):
    result_status = solver.Solve()
    if result_status != pywraplp.Solver.OPTIMAL:
        raise Exception("failed to solve")
    if not solver.VerifySolution(1e-7, True):
        raise Exception("solution is wonky")
    solution = {r: {c: None for c in range(9)} for r in range(9)}
    for row in range(9):
        for col in range(9):
            for num in range(9):
                if board[row][col][num].solution_value():
                    solution[row][col] = num + 1
    return solution


def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', metavar='INPUT', type=str, help='path to the given suduko problem')
    args = parser.parse_args(args)

    solver = pywraplp.Solver('Sudoku', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    givens = read_givens(args.input)
    board = prepare(solver, givens)
    solution = solve(solver, board)
    print_solution(solution)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
