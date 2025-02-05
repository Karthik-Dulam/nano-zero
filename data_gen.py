import json
from sudoku import Sudoku  # python -m pip install pysudoku
from tqdm import tqdm  # Optional progress bar

def generate_sudoku_dataset(num_samples=1000, difficulty=0.5):
    """
    Generate Sudoku dataset with puzzles and solutions
    Args:
        num_samples: Number of Sudoku puzzles to generate
        difficulty: Between 0 (easiest) and 1 (hardest)
    """
    dataset = []
    
    for _ in tqdm(range(num_samples)):
        # Generate puzzle
        puzzle = Sudoku(3).difficulty(difficulty)
        unsolved = puzzle.board
        
        # Solve puzzle
        solution = puzzle.solve()
        solved = solution.board
        
        # Convert to 81-digit strings
        puzzle_str = " ".join(
            str(cell or 0)
            for row in unsolved 
            for cell in row
        )
        
        solution_str = " ".join(
            str(cell or 0) for row in solved for cell in row
        )
        
        dataset.append({
            "puzzle": puzzle_str,
            "solution": solution_str
        })
    
    return dataset

# Generate and save dataset for each difficulty level (0.0, 0.5, 1.0)
difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_samples = 10_000

for difficulty in difficulties:
    dataset = generate_sudoku_dataset(num_samples=num_samples, difficulty=difficulty)
    filename = f"sudoku_dataset_difficulty_{difficulty}.json"
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} Sudoku puzzles for difficulty {difficulty}")