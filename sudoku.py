#!/usr/bin/env python3
"""A sudoku solver that uses mixed-integer programming."""

from ortools.linear_solver import pywraplp

import sys


def prepare(solver, givens):
    # The board is a 9x9 grid of cells.  Each cell has 9 boolean variables: one for each possible
    # number that could be placed in that cell.
    #
    # When solved, exactly one boolean variable per cell will be true, which represents the number
    # that is placed in that cell for the solution.
    board = {r: {c: {n: None for n in range(0, 9)} for c in range(0, 9)} for r in range(0, 9)}
    for row in range(0, 9):
        for col in range(0, 9):
            for num in range(0, 9):
                board[row][col][num] = solver.BoolVar('%d.%d=%d' % (row, col, num))
                # In case there are multiple solutions, each variable is given a unique objective
                # value so that the solution is deterministic.
                solver.Objective().SetCoefficient(board[row][col][num], 10 * (row * 9 + col) + num)

    # Rules of sudoku:
    #
    #  1. Each cell has exactly one number.
    #  2. Each number can only be present in each row once.
    #  3. Each number can only be present in each column once.
    #  4. Each number can only be present in each 3x3 sub-grid once.
    #
    # The following loops build up those constraints.

    # Each cell must have exactly one number set.
    for row in range(0, 9):
        for col in range(0, 9):
            solver.Add(sum(board[row][col].values()) == 1)

    for num in range(0, 9):
        # Each row must have each number set exactly once.
        for row in range(0, 9):
            solver.Add(sum(board[row][c][num] for c in range(0, 9)) == 1)

        # Each column must have each number set exactly once.
        for col in range(0, 9):
            solver.Add(sum(board[r][col][num] for r in range(0, 9)) == 1)

        # Each 3x3 square must have each number set exactly once.
        for i in range(0, 9):
            start_row = (i // 3) * 3
            start_col = (i % 3) * 3
            solver.Add(sum(board[start_row + r][start_col + c][num] for r in range(0, 3) for c in range(0, 3)) == 1)
    
    # Pin the cells that have given values to those values.
    for row in range(0, 9):
        for col in range(0, 9):
            if givens[row][col]:
                given_num = givens[row][col] - 1
                solver.Add(board[row][col][given_num] == 1)

    return board


def read_givens():
    givens = {r: {c: None for c in range(0, 9)} for r in range(0, 9)}

    for row, line in enumerate(sys.stdin):
        for col, value in enumerate(line.split()):
            try:
                givens[row][col] = int(value)
            except ValueError:
                pass
    
    return givens


def print_board( board):
    for row in range(0, 9):
        for col in range(0, 9):
            for num in range(0, 9):
                if board[row][col][num].solution_value():
                    print(' %d' % (num + 1), end='')
            if col in (2, 5):
                print(' |', end='')
        print()
        if row in (2, 5):
            print(' %s+%s+%s' % ('-' * 6, '-' * 7, '-' * 6))


def solve(solver):
    result_status = solver.Solve()
    if result_status != pywraplp.Solver.OPTIMAL:
        raise Exception("failed to solve")
    if not solver.VerifySolution(1e-7, True):
        raise Exception("solution is wonky")


def main(unused_args):
    solver = pywraplp.Solver('Sudoku', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    givens = read_givens()
    board = prepare(solver, givens)
    solve(solver)
    print_board(board)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
