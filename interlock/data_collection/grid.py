"""Choose a puzzle's grid for an image: draw a cell size, then crop the image to a whole number of cells.

Cell sizes come from CELL_MIN, CELL_MIN + CELL_STEP, ..., CELL_MAX (see cutting.py). That's the size before
cutting.GRID_JITTER: each piece's box is the cell plus PAD, the same square for every piece of a puzzle, and
the jittered piece varies inside it. The grid fills as much of the image as
fits; less than one cell is cropped from each dimension. Called with a seeded rng, the same seed always gives the same
grid and crop, so a new seed every epoch gives new puzzle sizes and a fixed seed gives a reproducible val set.
"""

import numpy as np

from interlock.data_collection.cutting import CELL_MAX, CELL_MIN, CELL_STEP

CELL_SIZES = np.arange(CELL_MIN, CELL_MAX + 1, CELL_STEP)
MIN_SIDE = 4  # fewest pieces along either side; smaller puzzles are mostly border pieces


def random_grid(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Draw a cell size and crop image to the largest whole grid of that size.

    Args:
        image: (H, W, 3) image.
        rng: random number generator; picks the cell size and where the crop sits.
    Returns:
        crop: (rows * cell, cols * cell, 3) view into image, ready for cut(crop, seed, rows, cols).
        rows, cols: grid size.
    """
    height, width = image.shape[:2]
    # only cell sizes that fit at least MIN_SIDE pieces along both sides of this image
    cell_sizes = CELL_SIZES[(height // CELL_SIZES >= MIN_SIDE) & (width // CELL_SIZES >= MIN_SIDE)]
    if len(cell_sizes) == 0:
        raise ValueError(f"{width}x{height} image is too small for a {MIN_SIDE}x{MIN_SIDE} grid of "
                         f"{CELL_MIN}px cells")

    cell = int(rng.choice(cell_sizes))
    rows, cols = height // cell, width // cell
    # the leftover (less than one cell) is split between the two sides at random, so no strip is always cut off
    top = int(rng.integers(0, height - rows * cell + 1))
    left = int(rng.integers(0, width - cols * cell + 1))
    return image[top : top + rows * cell, left : left + cols * cell], rows, cols
