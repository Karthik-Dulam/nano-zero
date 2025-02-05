import json
from sudoku import Sudoku
import numpy as np
from tqdm import tqdm

def generate_teaching_data(num_samples=100_000, difficulty=0.1):
    """
    Generate Sudoku puzzles with alternating row/column checks
    """
    dataset = []
    for _ in tqdm(range(num_samples)):
        puzzle = Sudoku(3, seed=np.random.randint(1000)).difficulty(difficulty)
        unsolved = [[cell or 0 for cell in row] for row in puzzle.board]
        solution = puzzle.solve().board

        puzzle_arr = np.array(unsolved)
        solution_arr = np.array(solution)
        
        steps = []
        temp_board = np.copy(puzzle_arr)
        check_row_first = True  # Alternation control flag
        
        while not np.all(temp_board != 0):
            found = False
            last_check = None

            # Alternating check pattern
            if check_row_first:
                # Check rows first
                for i in range(9):
                    row = temp_board[i]
                    if np.count_nonzero(row == 0) == 1:
                        missing = 45 - row.sum()
                        j = np.where(row == 0)[0][0]
                        steps.append(f"In row {i+1} the only missing element is {missing} so row {i+1} column {j+1} must be {missing}.")
                        temp_board[i][j] = missing
                        found = True
                        last_check = 'row'
                        break
                if not found:
                    # Then check columns
                    for j in range(9):
                        col = temp_board[:, j]
                        if np.count_nonzero(col == 0) == 1:
                            missing = 45 - col.sum()
                            i = np.where(col == 0)[0][0]
                            steps.append(f"In column {j+1} the only missing element is {missing} so row {i+1} column {j+1} must be {missing}.")
                            temp_board[i][j] = missing
                            found = True
                            last_check = 'column'
                            break
            else:
                # Check columns first
                for j in range(9):
                    col = temp_board[:, j]
                    if np.count_nonzero(col == 0) == 1:
                        missing = 45 - col.sum()
                        i = np.where(col == 0)[0][0]
                        steps.append(f"In column {j+1} the only missing element is {missing} so row {i+1} column {j+1} must be {missing}.")
                        temp_board[i][j] = missing
                        found = True
                        last_check = 'column'
                        break
                if not found:
                    # Then check rows
                    for i in range(9):
                        row = temp_board[i]
                        if np.count_nonzero(row == 0) == 1:
                            missing = 45 - row.sum()
                            j = np.where(row == 0)[0][0]
                            steps.append(f"In row {i+1} the only missing element is {missing} so row {i+1} column {j+1} must be {missing}.")
                            temp_board[i][j] = missing
                            found = True
                            last_check = 'row'
                            break

            # Switch check order for next iteration based on last found
            if last_check:
                check_row_first = (last_check == 'column')
            else:
                check_row_first = not check_row_first

            if found:
                continue

            # Check blocks if no row/column found
            for block_i in range(3):
                for block_j in range(3):
                    block = temp_board[block_i*3:(block_i+1)*3, block_j*3:(block_j+1)*3]
                    if np.count_nonzero(block == 0) == 1:
                        missing = 45 - block.sum()
                        idx = np.where(block == 0)
                        i = block_i*3 + idx[0][0]
                        j = block_j*3 + idx[1][0]
                        steps.append(f"In block ({block_i+1}, {block_j+1}) the only missing element is {missing} so row {i+1} column {j+1} must be {missing}.")
                        temp_board[i][j] = missing
                        found = True
                        break
                if found:
                    break

            if not found:
                break

        # Convert boards to strings
        puzzle_str = " ".join(map(str, puzzle_arr.flatten()))
        solution_str = " ".join(map(str, solution_arr.flatten()))
        
        # Format thinking text
        board_visual = "\n".join([" ".join(map(str, row)) for row in temp_board])
        thinking_text = (
            "<thonk> I see a sudoku problem. Most of its cells are filled. So it should be easy to finish it.\n"
            + "\n".join(steps) +
            "\nI think this completes the sudoku. Let me check:\n" + board_visual +
            "\nLets see if it satisfies the sudoku rules. </thonk>"
        )
        
        # Format final answer
        formatted_solution = "".join([str(n) for row in solution for n in row])
        answer_text = f"<ans>\n{formatted_solution}\n</ans>"
        
        dataset.append({
            # "puzzle": puzzle,
            "puzzle": puzzle_str,
            "solution": formatted_solution,
            "output": thinking_text + "\n" + answer_text
        })
    return dataset

if __name__ == "__main__":
    # Generate and save dataset
    dataset = generate_teaching_data(num_samples=100_000, difficulty=0.2)

    # Save in instruction format
    instruction_dataset = [{
        "instruction": f"Solve this Sudoku puzzle:",
        "input": example['puzzle'],
        "output": example['output']
    } for example in dataset]

    with open("sudoku_sft_data.json", "w") as f:
        json.dump(instruction_dataset, f, indent=2)

    print(f"Generated {len(dataset)} training examples")