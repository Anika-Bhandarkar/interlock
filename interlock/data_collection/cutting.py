"""Helper function to cut an image into jigsaw pieces. This is done by finding the corner points of each piece,
sampling points along the edge of each piece using randomized bezier curve to vary piece geometry, putting all edges of
a piece together into a boundary for that piece, then using boundaries as image masks to cut out individual pieces.

For now, puzzle size is hardcoded to be 10 x 10 for convenience. Later versions will support custom
puzzle sizes.

v1 bezier curve implementation borrowed from https://github.com/Draradech/jigsaw
"""

import numpy as np
from PIL import Image, ImageDraw

ROWS = 10
COLS = 10
# mimicking publicly available configuration for now.
TAB_SIZE = 0.2
JITTER = 0.04
SAMPLES_PER_SEGMENT = 20
PAD = 16 # margin around each cell in a piece's box; must exceed max tab reach (~0.29 * cell)

def cut(image: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut an image into a ROWS x COLS grid of jigsaw pieces.
    Precondition: image height is divisible by ROWS and image width is divisible by COLS.

    Args:
        image: (H, W, 3) uint8 image to cut into puzzle pieces.
        seed: random seed; the same seed always gives the same puzzle.

    Returns (N = ROWS * COLS, B = cell size + 2 * PAD):
        pieces: (N, B, B, 3) uint8. Each piece in a box of its cell plus PAD on every side; 0 outside
            the piece.
        masks: (N, B, B) bool. True where a pixel of the box belongs to the piece.
        positions: (N, 2) int. positions[i] = (x, y) grid coordinates of pieces[i], x = column and
            y = row, with (0, 0) the top left piece. Pieces are returned in grid order (not shuffled).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got dtype {image.dtype}")

    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    corners = get_corners(height, width)
    horizontal_edges, vertical_edges = get_edges(corners, rng)
    piece_ids = piece_map(horizontal_edges, vertical_edges, height, width)
    pieces, masks = extract_pieces(image, piece_ids)

    # (x, y) = (column, row) built directly; flipping with [:, ::-1] gives negative strides that torch rejects
    positions = np.array([(piece % COLS, piece // COLS) for piece in range(ROWS * COLS)])

    return pieces, masks, positions


def get_corners(height: int, width: int) -> np.ndarray:
    """Returns the coordinates of corners of every jigsaw piece.
    A puzzle with N x M jigsaw pieces will have (N + 1) x (M + 1) corners.

    Precondition: image height is divisible by ROWS and image width is divisible by COLS.
    Args:
        height: image height
        width: image width

    Return:
        corners: (ROWS + 1) x (COLS + 1) x 2 array of corner points. corners[i, j] = (x, y) is the
        pixel where horizontal grid line i (counting from the top) meets vertical grid line j
        (counting from the left). e.x.:
        [[(0, 0) (5, 0) (10, 0)], [(0, 5) (5, 5) (10, 5)], [(0, 10) (5, 10) (10, 10)]]
        for a 10 x 10 image cut into a 2 x 2 grid.
    """
    if height % ROWS != 0:
        raise ValueError("height must be divisible by number of rows")
    if width % COLS != 0:
        raise ValueError("width must be divisible by number of columns")

    # columns are x-values and rows are y-values.
    rows = np.arange(ROWS + 1) * (height // ROWS)
    cols = np.arange(COLS + 1) * (width // COLS)

    x_coords, y_coords = np.meshgrid(cols, rows)
    return np.stack((x_coords, y_coords), axis=2)


def tab_control_points(rng: np.random.Generator) -> np.ndarray:
    """(3, 4, 2) array: 3 cubic bezier segments, each containing 4 (x, y) control points.
    Control points are on the standard edge from (0, 0) to (1, 0). Tabs point in the +y direction;
    place_edge flips them as needed.

    v1 implementation borrowed from https://github.com/Draradech/jigsaw
    """

    t = TAB_SIZE / 2
    # randomly generated offsets to use later
    a, b, c, d, e = rng.uniform(-JITTER, JITTER, 5)

    p0, p9 = (0, 0), (1, 0) #first and last points must be endpoints of the edge
    p1, p8 = (0.2, a), (0.8, e) # points closest to edges; offsets a and e make shoulders non-straight
    p2, p7 = (0.5 + b + d, -t + c), (0.5 + b + d, -t + c) # middle of neck point. shared for symmetry.
    p3, p6 = (0.5 - t + b, t + c), (0.5 + t + b, t + c) # left/right sides of neck
    p4, p5 = (0.5 - 2 * t + b - d, 3*t + c), (0.5 + 2 * t + b - d, 3*t + c) # head handles. head must be wider than neck

    control_points = np.array([np.array([p0, p1, p2, p3]), np.array([p3, p4, p5, p6]), np.array([p6, p7, p8, p9])])
    return control_points


def sample_curve(segments: np.ndarray, n: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Samples points along the bezier curves specified by tab_control_points().

    Args:
        segments: (3, 4, 2) output of tab_control_points
        n: samples per segment
    Returns:
        points: (3n - 2, 2) array of points along the tab shape. n points per segment, minus the 2
        joints shared between segments.
    """

    sample_vals = np.linspace(0, 1, n, endpoint=True)

    def bezier_point(P: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Point on the cubic Bézier with control points P (4, 2) at parameter t."""
        return (1-t)**3 * P[0] + 3*(1-t)**2*t * P[1] + 3*(1-t)*t**2 * P[2] + t**3 * P[3]

    # (n, 1) column so each t broadcasts against both x and y of each control point
    t = sample_vals[:, None]
    curves = [bezier_point(seg, t) for seg in segments]  # three (n, 2) arrays

    # each segment after the first starts on the previous segment's last point; drop the repeat
    return np.concatenate([curves[0]] + [curve[1:] for curve in curves[1:]])


def place_edge(curve: np.ndarray, start: np.ndarray, end: np.ndarray, sign: int) -> np.ndarray:
    """Takes a tab drawn on standard edge and maps it to one real edge of the puzzle.
    Inputs:
        curve: (K, 2) output of sample_curve
        start: (2,) starting corner of the edge. in (x, y) coordinates.
        end: (2, ) ending corner of the edge. in (x, y) coordinates.
        sign: +1, -1, 0 - which side the tab goes on.
            +1: TAB EXTENDS TO PIECE BELOW OR LEFT
            -1: TAB EXTENDS TO PIECE ABOVE OR RIGHT
             0: straight edge
            (assumes horizontal edges run left to right and vertical edges top to bottom)
    Returns:
        edge: (K, 2) - curve placed along a real edge.
    For some standardized (u, v), u represents how far along the edge our point is and v represents
    how far "out" (perpendicular distance) the point is. So:
        1. scale u by vector in direction of end - start
        2. scale v by vector perpendicular to end - start
    """
    if sign not in [-1, 0, 1]:
        raise ValueError("sign must be -1, 0, or 1.")

    direction = end - start # vector in the direction of travel
    perp = (-direction[1], direction[0]) # vector perpendicular to direction of travel
    u = curve[:, :1]                  # (K, 1): how far along, as a column
    v = curve[:, 1:]                  # (K, 1): how far out, as a column
    points = start + u * direction + sign * v * perp   # (K, 2)

    return points

def get_edges(corners: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Takes in the puzzle's corners and returns sampled points along every jigsaw edge.
    Border edges are straight; inner edges get a random tab on a random side.
    Input:
        corners: (ROWS + 1, COLS + 1, 2) output of get_corners
        rng: random number generator (created once in cut)
    Returns:
        horizontal_edges: all horizontal edges (left to right). (ROWS + 1, COLS, K, 2)
        vertical_edges: all vertical edges (top to bottom). (ROWS, COLS + 1, K, 2)
    where K = 3 * SAMPLES_PER_SEGMENT - 2
    """

    num_samples = 3 * SAMPLES_PER_SEGMENT - 2
    horizontal_edges = np.empty((ROWS + 1, COLS, num_samples, 2))
    vertical_edges = np.empty((ROWS, COLS + 1, num_samples, 2))

    def make_edge(start: np.ndarray, end: np.ndarray, is_border: bool) -> np.ndarray:
        # border edges must be straight; inner edges get a tab on a random side.
        curve = sample_curve(tab_control_points(rng))
        sign = 0 if is_border else rng.choice([-1, 1])
        return place_edge(curve, start, end, sign)

    #horizontals: line i runs left to right from corners[i, j] to corners[i, j + 1]
    for i in range(ROWS + 1):
        for j in range(COLS):
            is_border = (i == 0 or i == ROWS)
            horizontal_edges[i, j] = make_edge(corners[i, j], corners[i, j + 1], is_border)

    #verticals: line j runs top to bottom from corners[i, j] to corners[i + 1, j]
    for i in range(ROWS):
        for j in range(COLS + 1):
            is_border = j == 0 or j == COLS
            vertical_edges[i, j] = make_edge(corners[i, j], corners[i + 1, j], is_border)

    return horizontal_edges, vertical_edges

def piece_outline(horizontal_edges: np.ndarray, vertical_edges: np.ndarray, row: int, col: int) -> np.ndarray:
    """Returns the closed outline of piece (row, col).
    Joins the piece's top, right, bottom and left edges clockwise into one continuous walk (bottom and
    left reversed), dropping repeated corner points.

    Args:
        horizontal_edges: (ROWS + 1, COLS, K, 2)
        vertical_edges: (ROWS, COLS + 1, K, 2)
        row: row of the piece to compute
        col: column of the piece to compute
        Note (row, col) rather than (x, y).
    Return:
        boundary: (4K-4, 2) array of points corresponding to the jigsaw piece's edge.

    """
    top = horizontal_edges[row,col]
    right = vertical_edges[row, col + 1]
    bottom = horizontal_edges[row+1, col][::-1] #reverse the bottom to go right to left
    left = vertical_edges[row, col][::-1] # reverse to go bottom to top
    boundary = np.concatenate([top, right[1:], bottom[1:], left[1:-1]])
    return boundary

def piece_map(horizontal_edges: np.ndarray, vertical_edges: np.ndarray, height: int, width: int) -> np.ndarray:
    """Returns a piece map assigning every pixel to an ID indicating what piece it belongs to.

    Args:
        horizontal_edges: (ROWS + 1, COLS, K, 2) array of horizontal edges in the puzzle
        vertical_edges: (ROWS, COLS + 1, K, 2) array of vertical edges in the puzzle
        height: total height of the puzzle, in pixels
        width: total width of the puzzle, in pixels
    Returns:
        piece_ids: (height, width) int map the same size as the image, where each pixel holds the ID
        of the piece it belongs to: row * COLS + col.
    """

    canvas = Image.new("I", (width, height), -1) #blank canvas, every pixel -1
    draw = ImageDraw.Draw(canvas) #tool to draw and fill outlines
    for i in range(ROWS):
        for j in range(COLS):
            label = i * COLS + j
            boundary = piece_outline(horizontal_edges, vertical_edges, i, j)
            draw.polygon(boundary.ravel().tolist(), fill=label) #expects a flattened list
    #convert back to numpy
    piece_ids = np.array(canvas)
    # every pixel should belong to some piece; leftover -1s mean an outline didn't close
    if (piece_ids == -1).any():
        raise RuntimeError("piece map has unassigned pixels; an outline may not be closed")
    return piece_ids


def extract_pieces(image: np.ndarray, piece_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cuts the image into pieces using the piece map. Each piece is returned in a box of its cell
    plus PAD on every side, so every piece has the same shape (B = cell size + 2 * PAD).

    Precondition: all cells are the same size (no jitter was added to the corner coordinates).

    Args:
        image: (H, W, 3) uint8 image
        piece_ids: (H, W) output of piece_map
    Returns:
        pieces: (N, B, B, 3) uint8; pieces[k] is the piece with ID k. 0 outside the piece.
        masks: (N, B, B) bool; True where a pixel of the box belongs to the piece.
    """

    height, width = piece_ids.shape
    cell_h, cell_w = height // ROWS, width // COLS
    box_h, box_w = cell_h + 2 * PAD, cell_w + 2 * PAD # B: cell plus PAD on every side
    num_pieces = ROWS * COLS

    # start from zeros so everything outside a piece is already 0
    pieces = np.zeros((num_pieces, box_h, box_w, 3), dtype=np.uint8)
    masks = np.zeros((num_pieces, box_h, box_w), dtype=bool)

    # pad edges of image so entire bounding box of edge pieces lies within the image
    padded_image = np.pad(image, ((PAD, PAD), (PAD, PAD), (0, 0))) # pads with 0
    padded_ids = np.pad(piece_ids, PAD, constant_values=-1) #pad with -1 so these aren't assigned to a piece

    for piece_index in range(ROWS * COLS):
        row, col = piece_index // COLS, piece_index % COLS
        # with padding around image, box starts exactly at the cell's original position
        top, left = row * cell_h, col * cell_w
        box_ids = padded_ids[top:top + box_h, left:left + box_w]
        box_pixels = padded_image[top:top + box_h, left:left + box_w]

        mask = (box_ids == piece_index) # drops neighbors' slivers and the -1 padding
        masks[piece_index] = mask
        pieces[piece_index][mask] = box_pixels[mask]

    return pieces, masks
