"""Metadata-only adapter for coordinate matrices, never an expression transform."""
from ..data.store import EmbeddingView
from ..integrity import canonical_hash


def training_embedding(parent):
    """Preserve supplied names; synthesize positional IDs only when absent."""
    d = parent.values.shape[1]
    supplied = 'coordinate_names' in parent.metadata
    names = parent.metadata.get('coordinate_names') if supplied else [f'component_{i+1}' for i in range(d)]
    if (not isinstance(names, (list, tuple)) or len(names) != d
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != d):
        raise ValueError('Coordinate IDs must be unique nonempty strings, one per column')
    adapted = EmbeddingView(parent.values, parent.cell_ids, dict(parent.metadata, coordinate_names=list(names)))
    if adapted.parent_reference() != parent.parent_reference():
        raise ValueError('Metadata adapter changed numerical input identity')
    return adapted, {
        'coordinate_ids_source': 'existing_metadata' if supplied else 'generated_one_based_column_positions',
        'coordinate_ids_sha256': canonical_hash(list(names)), 'dimension': d,
        'gene_names_inferred': False, 'values_and_row_column_order_unchanged': True,
        'stored_metadata_overwritten': False,
    }
