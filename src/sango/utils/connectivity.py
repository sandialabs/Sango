# General Imports
import numpy as np

# Convert adjacency list to tuples (CSR)
def from_adjcy_list(adjcy, order='source'):
    """Convert an adjacency-list representation to ``[(source, target), ...]``.

    Allows for list of lists: [[1, 2], [], [0]]
    Allows for dict of lists: {0: [1, 2], 2: [0]}
    These get transformed to: [(0, 1), (0, 2), (2, 0)]
    Source-major ordering by default.
    """
    edges = []

    if isinstance(adjcy, dict):
        rows = adjcy.items()
    else:
        rows = enumerate(adjcy)

    if order == 'source':
        for source, targets in rows:
            for target in targets:
                edges.append((int(source), int(target)))
    else: # order == 'target'
        for target, sources in rows:
            for source in sources:
                edges.append((int(source), int(target)))

    return edges


# Convert dense adjacency matrix to tuples
def from_adjcy_matrix(matrix, order='source'):
    """Convert a dense adjacency matrix to ``[(source, target), ...]``.

    Nonzero entries are treated as edges.
    By default: rows are sources, columns are targets.
    """
    arr = np.asarray(matrix)

    if arr.ndim != 2:
        raise ValueError(f"adjacency matrix must be 2-D, got {arr.ndim}-D")

    if order == 'source':
        source_index, target_index = np.nonzero(arr)
    else: # order == 'target'
        target_index, source_index = np.nonzero(arr)

    return [(int(source), int(target))
            for source, target in zip(source_index, target_index)]


# Convert sparse adjacency matrix to tuples
def from_adjcy_sparse(sparse, order='source'):
    """Convert a scipy-style sparse adjacency matrix to ``[(source, target), ...]``.

    This converts the matrix to coo format first to get rows and cols.
    """
    if not hasattr(sparse, "tocoo"):
        raise TypeError("sparse adjacency must provide a tocoo() method")

    # Convert to COO format if not already
    if not(hasattr(sparse, "row") and hasattr(sparse, "col")):
        sparse = sparse.tocoo()

    if order == 'source':
        source_index = np.asarray(sparse.row, dtype=int)
        target_index = np.asarray(sparse.col, dtype=int)
    else: # order == 'target'
        target_index = np.asarray(sparse.row, dtype=int)
        source_index = np.asarray(sparse.col, dtype=int)

    return [(int(source), int(target))
            for source, target in zip(source_index, target_index)]


# Wrapper around specific adjacency conversion methods
def from_adjcy(adjcy, order='source', format='auto'):
    """Convert an adjacency representation to list of edge tuples."""
    if order not in ('source', 'target'):
        raise ValueError("order must be either 'source' or 'target'")

    if format == 'auto':
        format = _infer_adjcy_format(adjcy)

    if format == 'list':
        return from_adjcy_list(adjcy, order=order)

    if format in ('matrix', 'dense'):
        return from_adjcy_matrix(adjcy, order=order)

    if format == 'sparse':
        return from_adjcy_sparse(adjcy, order=order)

    raise ValueError(
        "format must be one of 'auto', 'list', 'matrix'/'dense', or 'sparse'"
    )


# Try to infer the adjacency format automatically
def _infer_adjcy_format(adjcy):
    """Infer which adjacency conversion method to use."""
    # Scipy sparse matrix
    if hasattr(adjcy, 'tocoo'):
        return 'sparse'

    # Adjacency list as dict of lists
    if isinstance(adjcy, dict):
        return 'list'

    # Numpy arrays
    if isinstance(adjcy, np.ndarray):
        if adjcy.ndim == 2:
            return 'matrix'
        raise ValueError(
            f"numpy adjacency input must be 2-D, got {adjcy.ndim}-D"
        )

    # Matrix-like non-list objects, e.g. pandas DataFrame.
    if hasattr(adjcy, 'ndim') and hasattr(adjcy, 'shape'):
        if adjcy.ndim == 2:
            return 'matrix'

    # Default to adjacency lists
    if isinstance(adjcy, (list, tuple)):
        return 'list'

    raise TypeError(
        "could not infer adjacency format; expected an adjacency list, "
        "a 2-D dense matrix, or a scipy-style sparse matrix with tocoo(). "
        "Pass format = 'list', 'matrix'/'dense', or 'sparse' explicitly."
    )
