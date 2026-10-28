"""Helper function to cut an image into jigsaw pieces. This is done by finding the corner points of each piece,
sampling points along the edge of each piece using randomized bezier curve to vary piece geometry, putting all edges of
a piece together into a boundary for that piece, then using boundaries as image masks to cut out individual pieces.

Each puzzle first draws a small set of tab shapes (tab_shapes), from textbook to funky, and every inner edge uses one
of them, so many edges share a shape and the model can't match pieces by shape alone.

Puzzles are ROWS x COLS (10 x 10) by default; pass rows and cols to cut() for other grid sizes.

v1 bezier curve implementation borrowed from https://github.com/Draradech/jigsaw
"""

import numpy as np
from PIL import Image, ImageDraw

# default grid size, used when cut() isn't given rows and cols
ROWS = 10
COLS = 10
SAMPLES_PER_SEGMENT = 20
# tab shapes: a shape is 6 numbers (t, a, b, c, d, e), see tab_control_points. TAB_NOMINAL is the textbook tab;
# a shape with funkiness s in [0, 1] moves each number away from it by up to s * TAB_MAX_DEV.
NUM_TAB_SHAPES = 6  # shapes per puzzle
TAB_NOMINAL = {"t": 0.105, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
TAB_MAX_DEV = {"t": 0.025, "a": 0.11, "b": 0.17, "c": 0.10, "d": 0.10, "e": 0.11}
TAB_MIN_DEV = {"t": 0.01}  # even the most regular tabs vary a little in size
DEV_FLOOR = 0.5  # every deviation is at least this fraction of s * TAB_MAX_DEV, so a high s is visibly funky
PLACEMENT_LIMIT = 1 / 6  # neck and head stay in the middle third of the edge, 0.5 +- 1/6
SHOULDER_MARGIN = 0.02  # smallest gap between the head and the shoulder points
MAX_HEAD_WIDTH = 0.46  # widest head (4t), as a fraction of the edge; wide heads crowd the piece's corners
# pairs.side_types tells tabs from blanks by probing SIDE_PROBE = 0.15 of a cell past where the edge would be without
# GRID_JITTER, so every tab must clear that, and the S-curve dip on the other side stay short of it, even after the edge
# moves by up to GRID_JITTER
MIN_TAB_HEIGHT = 0.22
MAX_DIP = 0.08
# each inner grid line is shifted by up to GRID_JITTER of a cell, so cells (and pieces) vary in width and height by up
# to 2 * GRID_JITTER. Larger jitter needs a side_types that finds the real edge: at 0.04, 30px cells start getting misread
GRID_JITTER = 0.0 # 0 for now. TODO: downstream scripts need to not assume that bounding box sits in the middle of piece pre-padding.
# supported cell sizes, in pixels: CELL_MIN, CELL_MIN + CELL_STEP, ..., CELL_MAX
CELL_MIN = 30
CELL_MAX = 100
CELL_STEP = 5
PAD = 42  # margin around each cell in a piece's box; must exceed max tab reach plus grid jitter (~0.4 * cell), 41 px at CELL_MAX
# every model input is padded to this side, the box of a CELL_MAX piece; cut() itself returns tighter cell + 2 * PAD boxes
PIECE_SIZE = CELL_MAX + 2 * PAD

def cut(image: np.ndarray, seed: int, rows: int = ROWS, cols: int = COLS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut an image into a rows x cols grid of jigsaw pieces.
    Precondition: image height is divisible by rows and image width is divisible by cols.

    Args:
        image: (H, W, 3) uint8 image to cut into puzzle pieces.
        seed: random seed; the same seed always gives the same puzzle.
        rows: number of pieces down the puzzle.
        cols: number of pieces across the puzzle.

    Returns (N = rows * cols, B_h = H / rows + 2 * PAD, B_w = W / cols + 2 * PAD; B_h = B_w when the
    cells are square):
        pieces: (N, B_h, B_w, 3) uint8. Each piece in a box of its unjittered cell plus PAD on every side; 0 outside
            the piece.
        masks: (N, B_h, B_w) bool. True where a pixel of the box belongs to the piece.
        positions: (N, 2) int. positions[i] = (x, y) grid coordinates of pieces[i], x = column and
            y = row, with (0, 0) the top left piece. Pieces are returned in grid order (not shuffled).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got dtype {image.dtype}")

    height, width = image.shape[:2]
    horizontal_edges, vertical_edges = puzzle_edges(height, width, np.random.default_rng(seed), rows, cols)
    piece_ids = piece_map(horizontal_edges, vertical_edges, height, width)
    pieces, masks = extract_pieces(image, piece_ids, rows, cols)

    # (x, y) = (column, row) built directly; flipping with [:, ::-1] gives negative strides that torch rejects
    positions = np.array([(piece % cols, piece // cols) for piece in range(rows * cols)])

    return pieces, masks, positions


def puzzle_edges(height: int, width: int, rng: np.random.Generator, rows: int = ROWS,
                 cols: int = COLS) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of one puzzle, drawn from rng in a fixed order: the set of tab shapes, then the jittered grid,
    then each edge's shape, mirror and side. cut() uses this, so the same rng seed always gives the same edges.

    Returns:
        horizontal_edges, vertical_edges: as returned by get_edges
    """
    shapes = tab_shapes(rng)
    corners = get_corners(height, width, rows, cols, rng)
    return get_edges(corners, rng, shapes)


def get_corners(height: int, width: int, rows: int = ROWS, cols: int = COLS, rng: np.random.Generator | None = None,
                jitter: float = GRID_JITTER) -> np.ndarray:
    """Returns the coordinates of corners of every jigsaw piece.
    A puzzle with N x M jigsaw pieces will have (N + 1) x (M + 1) corners.

    Grid lines are evenly spaced, then (if rng is given) each inner line is shifted by up to jitter * cell, so
    cells vary in size. The border lines stay on the image border, and grid lines stay straight, so every piece
    in a column has the same width and every piece in a row the same height.

    Precondition: image height is divisible by rows and image width is divisible by cols.
    Args:
        height: image height
        width: image width
        rows: number of pieces down the puzzle
        cols: number of pieces across the puzzle
        rng: random number generator for the jitter; None for an exact grid
        jitter: largest shift of an inner line, as a fraction of a cell

    Return:
        corners: (rows + 1) x (cols + 1) x 2 array of corner points. corners[i, j] = (x, y) is the
        pixel where horizontal grid line i (counting from the top) meets vertical grid line j
        (counting from the left). e.x.:
        [[(0, 0) (5, 0) (10, 0)], [(0, 5) (5, 5) (10, 5)], [(0, 10) (5, 10) (10, 10)]]
        for a 10 x 10 image cut into a 2 x 2 grid.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be at least 1, got {rows} x {cols}")
    if height % rows != 0:
        raise ValueError(f"height {height} must be divisible by number of rows {rows}")
    if width % cols != 0:
        raise ValueError(f"width {width} must be divisible by number of columns {cols}")

    # columns are x-values and rows are y-values.
    cell_h, cell_w = height // rows, width // cols
    row_lines = np.arange(rows + 1) * cell_h
    col_lines = np.arange(cols + 1) * cell_w
    if rng is not None:
        row_lines = row_lines + np.r_[0, rng.uniform(-jitter, jitter, rows - 1) * cell_h, 0]
        col_lines = col_lines + np.r_[0, rng.uniform(-jitter, jitter, cols - 1) * cell_w, 0]

    x_coords, y_coords = np.meshgrid(col_lines, row_lines)
    return np.stack((x_coords, y_coords), axis=2)


def tab_shapes(rng: np.random.Generator, num_shapes: int = NUM_TAB_SHAPES) -> np.ndarray:
    """The set of tab shapes one puzzle's inner edges are drawn from, sorted from regular to funky.

    Shape k gets funkiness s drawn from the k-th of num_shapes equal slots of [0, 1], so every puzzle mixes textbook
    tabs (s near 0) with funky ones (s near 1). A shape that fails is_valid_tab is redrawn at the same s; drawing a
    new s instead would skew the set toward regular shapes, which fail less often.

    Args:
        rng: random number generator (created once in cut)
        num_shapes: how many shapes in the set
    Returns:
        shapes: (num_shapes, 3, 4, 2) control points, each as returned by tab_control_points
    """
    funkiness = (np.arange(num_shapes) + rng.uniform(size=num_shapes)) / num_shapes
    shapes = []
    for s in funkiness:
        params = tab_params(rng, s)
        while not is_valid_tab(*params):
            params = tab_params(rng, s)
        shapes.append(tab_control_points(*params))
    return np.stack(shapes)


def tab_params(rng: np.random.Generator, funkiness: float) -> tuple[float, ...]:
    """(t, a, b, c, d, e) for one shape: TAB_NOMINAL plus, for each number, a deviation with a random sign and a
    size between DEV_FLOOR and 1 times funkiness * TAB_MAX_DEV (never under TAB_MIN_DEV)."""
    params = []
    for name in "tabcde":
        scale = max(funkiness * TAB_MAX_DEV[name], TAB_MIN_DEV.get(name, 0.0))
        params.append(TAB_NOMINAL[name] + rng.choice([-1, 1]) * rng.uniform(DEV_FLOOR, 1) * scale)
    return tuple(params)


def is_valid_tab(t: float, a: float, b: float, c: float, d: float, e: float) -> bool:
    """True if the shape stays in the middle third, its head isn't too wide and clears the shoulders, it sticks out
    far enough (and dips back little enough) for side_types to read it, and its outline doesn't cross itself."""
    in_middle_third = abs(b) + abs(d) <= PLACEMENT_LIMIT  # neck centre is at 0.5 + b + d, head centre at 0.5 + b - d
    # the head reaches 2t either side of its centre; the shoulder points sit 0.3 either side of 0.5 + b
    head_clears_shoulders = 2 * t + abs(d) <= 0.3 - SHOULDER_MARGIN
    if not (in_middle_third and head_clears_shoulders and 4 * t <= MAX_HEAD_WIDTH):
        return False
    curve = sample_curve(tab_control_points(t, a, b, c, d, e))
    readable = curve[:, 1].max() >= MIN_TAB_HEIGHT and -curve[:, 1].min() <= MAX_DIP
    return readable and not self_intersects(curve)


def tab_control_points(t: float, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """(3, 4, 2) array: 3 cubic bezier segments, each containing 4 (x, y) control points.
    Control points are on the standard edge from (0, 0) to (1, 0). Tabs point in the +y direction;
    place_edge flips them as needed.

    Args:
        t: tab size (the head is 4t wide)
        a, e: left and right shoulder wobble
        b: placement along the edge; the shoulders move with the tab so an off-centre tab keeps its S-curve
        c: how far the tab sticks out
        d: lean; the neck moves by +d and the head by -d

    v1 implementation borrowed from https://github.com/Draradech/jigsaw
    """
    p0, p9 = (0, 0), (1, 0) #first and last points must be endpoints of the edge
    p1, p8 = (0.2 + b, a), (0.8 + b, e) # points closest to edges; offsets a and e make shoulders non-straight
    p2, p7 = (0.5 + b + d, -t + c), (0.5 + b + d, -t + c) # middle of neck point. shared for symmetry.
    p3, p6 = (0.5 - t + b, t + c), (0.5 + t + b, t + c) # left/right sides of neck
    p4, p5 = (0.5 - 2 * t + b - d, 3*t + c), (0.5 + 2 * t + b - d, 3*t + c) # head handles. head must be wider than neck

    return np.array([[p0, p1, p2, p3], [p3, p4, p5, p6], [p6, p7, p8, p9]], dtype=float)


def self_intersects(curve: np.ndarray) -> bool:
    """True if any two non-adjacent segments of the (K, 2) polyline cross."""
    def turn(origin, a, b):  # z of (a - origin) x (b - origin): which side of origin -> a the point b is on
        return (a[..., 0] - origin[..., 0]) * (b[..., 1] - origin[..., 1]) - (a[..., 1] - origin[..., 1]) * (b[..., 0] - origin[..., 0])

    p, q = curve[:-1, None], curve[1:, None]  # segment i, as a column
    r, s = curve[None, :-1], curve[None, 1:]  # segment j, as a row
    crosses = (turn(r, s, p) * turn(r, s, q) < 0) & (turn(p, q, r) * turn(p, q, s) < 0)
    return bool(np.triu(crosses, k=2).any())  # k=2 skips each segment and its neighbor, which share a point


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

def get_edges(corners: np.ndarray, rng: np.random.Generator, shapes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Takes in the puzzle's corners and returns sampled points along every jigsaw edge.
    Border edges are straight; each inner edge gets a random shape from the puzzle's set, mirrored or not,
    with the tab on a random side.
    Input:
        corners: (rows + 1, cols + 1, 2) output of get_corners; the grid size is read from its shape
        rng: random number generator (created once in cut)
        shapes: (S, 3, 4, 2) set of tab shapes; drawn with tab_shapes(rng) first thing if not given
    Returns:
        horizontal_edges: all horizontal edges (left to right). (rows + 1, cols, K, 2)
        vertical_edges: all vertical edges (top to bottom). (rows, cols + 1, K, 2)
    where K = 3 * SAMPLES_PER_SEGMENT - 2
    """

    rows, cols = corners.shape[0] - 1, corners.shape[1] - 1
    num_samples = 3 * SAMPLES_PER_SEGMENT - 2
    horizontal_edges = np.empty((rows + 1, cols, num_samples, 2))
    vertical_edges = np.empty((rows, cols + 1, num_samples, 2))

    if shapes is None:
        shapes = tab_shapes(rng)
    curves = [sample_curve(shape) for shape in shapes]

    def make_edge(start: np.ndarray, end: np.ndarray, is_border: bool) -> np.ndarray:
        # border edges must be straight (sign 0 flattens any curve); inner edges get a tab on a random side.
        if is_border:
            return place_edge(curves[0], start, end, 0)
        curve = curves[rng.integers(len(curves))]
        if rng.random() < 0.5:  # mirror along the edge, still running from (0, 0) to (1, 0)
            curve = np.column_stack([1 - curve[::-1, 0], curve[::-1, 1]])
        return place_edge(curve, start, end, rng.choice([-1, 1]))

    #horizontals: line i runs left to right from corners[i, j] to corners[i, j + 1]
    for i in range(rows + 1):
        for j in range(cols):
            is_border = (i == 0 or i == rows)
            horizontal_edges[i, j] = make_edge(corners[i, j], corners[i, j + 1], is_border)

    #verticals: line j runs top to bottom from corners[i, j] to corners[i + 1, j]
    for i in range(rows):
        for j in range(cols + 1):
            is_border = j == 0 or j == cols
            vertical_edges[i, j] = make_edge(corners[i, j], corners[i + 1, j], is_border)

    return horizontal_edges, vertical_edges

def piece_outline(horizontal_edges: np.ndarray, vertical_edges: np.ndarray, row: int, col: int) -> np.ndarray:
    """Returns the closed outline of piece (row, col).
    Joins the piece's top, right, bottom and left edges clockwise into one continuous walk (bottom and
    left reversed), dropping repeated corner points.

    Args:
        horizontal_edges: (rows + 1, cols, K, 2)
        vertical_edges: (rows, cols + 1, K, 2)
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
        horizontal_edges: (rows + 1, cols, K, 2) array of horizontal edges in the puzzle
        vertical_edges: (rows, cols + 1, K, 2) array of vertical edges in the puzzle
        height: total height of the puzzle, in pixels
        width: total width of the puzzle, in pixels
    Returns:
        piece_ids: (height, width) int map the same size as the image, where each pixel holds the ID
        of the piece it belongs to: row * cols + col.
    """

    rows, cols = vertical_edges.shape[0], horizontal_edges.shape[1]
    canvas = Image.new("I", (width, height), -1) #blank canvas, every pixel -1
    draw = ImageDraw.Draw(canvas) #tool to draw and fill outlines
    for i in range(rows):
        for j in range(cols):
            label = i * cols + j
            boundary = piece_outline(horizontal_edges, vertical_edges, i, j)
            draw.polygon(boundary.ravel().tolist(), fill=label) #expects a flattened list
    #convert back to numpy
    piece_ids = np.array(canvas)
    # every pixel should belong to some piece; leftover -1s mean an outline didn't close
    if (piece_ids == -1).any():
        raise RuntimeError("piece map has unassigned pixels; an outline may not be closed")
    return piece_ids


def extract_pieces(image: np.ndarray, piece_ids: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Cuts the image into pieces using the piece map. Each piece is returned in a box of its cell
    plus PAD on every side, so every piece has the same shape (B = cell size + 2 * PAD).

    Boxes sit on the grid without jitter (cell = H / rows), so a box keeps the same place relative to its neighbors'
    boxes and true neighbors line up exactly; PAD covers the jitter as well as the tabs.

    Args:
        image: (H, W, 3) uint8 image
        piece_ids: (H, W) output of piece_map
        rows, cols: grid size the piece map was built with
    Returns:
        pieces: (N, B_h, B_w, 3) uint8; pieces[k] is the piece with ID k. 0 outside the piece.
        masks: (N, B_h, B_w) bool; True where a pixel of the box belongs to the piece.
    """

    height, width = piece_ids.shape
    cell_h, cell_w = height // rows, width // cols
    box_h, box_w = cell_h + 2 * PAD, cell_w + 2 * PAD # B: cell plus PAD on every side
    num_pieces = rows * cols

    # start from zeros so everything outside a piece is already 0
    pieces = np.zeros((num_pieces, box_h, box_w, 3), dtype=np.uint8)
    masks = np.zeros((num_pieces, box_h, box_w), dtype=bool)

    # pad edges of image so entire bounding box of edge pieces lies within the image
    padded_image = np.pad(image, ((PAD, PAD), (PAD, PAD), (0, 0))) # pads with 0
    padded_ids = np.pad(piece_ids, PAD, constant_values=-1) #pad with -1 so these aren't assigned to a piece

    for piece_index in range(num_pieces):
        row, col = piece_index // cols, piece_index % cols
        # with padding around image, box starts exactly at the cell's original position
        top, left = row * cell_h, col * cell_w
        box_ids = padded_ids[top:top + box_h, left:left + box_w]
        box_pixels = padded_image[top:top + box_h, left:left + box_w]

        mask = (box_ids == piece_index) # drops neighbors' slivers and the -1 padding
        masks[piece_index] = mask
        pieces[piece_index][mask] = box_pixels[mask]

    return pieces, masks
