# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database/_faiss_utils.py
"""
FAISS utility functions for LocalVectorDB.

This module centralizes FAISS-specific operations to handle version
compatibility, ID mapping access, and provide consistent fallback
behavior across the codebase.

The FAISS library occasionally changes internal APIs, and different
index types expose different attributes. This module provides a
stable interface that gracefully handles these variations.
"""

import logging
from typing import Any, Dict, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def safe_get_id_mapping(index: Any) -> Optional[np.ndarray]:
    """
    Safely extract external ID mapping from a FAISS index.

    This function handles various FAISS index types and versions by attempting
    multiple access patterns with proper fallbacks. It's designed to work with
    IndexIDMap, IndexIDMap2, and other ID-mapped index types.

    Parameters
    ----------
    index : Any
        A FAISS index object, potentially with ID mapping capabilities.

    Returns
    -------
    Optional[np.ndarray]
        Array of external IDs if available, None if not accessible or
        if the index doesn't support ID mapping.

    Notes
    -----
    Different FAISS versions and index types expose ID mappings differently:
    - IndexIDMap2: Often has .id_map.id_map attribute
    - IndexIDMap: May have different internal structure
    - Some versions use different attribute names

    This function tries multiple approaches and logs the method used
    for debugging purposes.

    Examples
    --------
    >>> import faiss
    >>> index = faiss.IndexIDMap2(faiss.IndexFlatL2(128))
    >>> external_ids = safe_get_id_mapping(index)
    >>> if external_ids is not None:
    ...     print(f"Found {len(external_ids)} external IDs")
    """
    if index is None:
        return None

    try:
        # Method 1: Standard IndexIDMap2 access pattern
        if hasattr(index, "id_map") and hasattr(index.id_map, "id_map"):
            logger.debug("Using IndexIDMap2 standard access (.id_map.id_map)")
            external_ids = faiss.vector_to_array(index.id_map.id_map).astype(np.int64)
            return external_ids

    except Exception as e:
        logger.debug(f"Standard IndexIDMap2 access failed: {e}")

    try:
        # Method 2: Direct id_map access (some FAISS versions)
        if hasattr(index, "id_map"):
            logger.debug("Attempting direct id_map access")
            # Try to access as vector-like object
            if hasattr(index.id_map, "size") and callable(getattr(index.id_map, "size", None)):
                size = index.id_map.size()
                if size > 0:
                    external_ids = faiss.vector_to_array(index.id_map).astype(np.int64)
                    return external_ids

    except Exception as e:
        logger.debug(f"Direct id_map access failed: {e}")

    try:
        # Method 3: Alternative attribute names (version compatibility)
        for attr_name in ["external_ids", "ids", "id_vector"]:
            if hasattr(index, attr_name):
                logger.debug(f"Using alternative attribute: {attr_name}")
                attr = getattr(index, attr_name)
                if hasattr(attr, "__len__") and len(attr) > 0:
                    return np.array(attr, dtype=np.int64)

    except Exception as e:
        logger.debug(f"Alternative attribute access failed: {e}")

    # Method 4: Check if it's a wrapped index
    try:
        if hasattr(index, "index") and hasattr(index.index, "id_map"):
            logger.debug("Attempting wrapped index access")
            return safe_get_id_mapping(index.index)

    except Exception as e:
        logger.debug(f"Wrapped index access failed: {e}")

    logger.debug(f"No ID mapping accessible for index type: {type(index)}")
    return None


def get_faiss_external_ids(index: Any) -> np.ndarray:
    """
    Get external IDs from a FAISS index with comprehensive error handling.

    This is a higher-level function that builds on safe_get_id_mapping()
    to provide additional context and error reporting.

    Parameters
    ----------
    index : Any
        A FAISS index object with ID mapping capabilities.

    Returns
    -------
    np.ndarray
        Array of external IDs.

    Raises
    ------
    ValueError
        If the index doesn't support ID mapping or if no IDs are available.
    RuntimeError
        If there's an unexpected error accessing the ID mapping.

    Examples
    --------
    >>> external_ids = get_faiss_external_ids(index)
    >>> print(f"Index contains {len(external_ids)} vectors")
    """
    if index is None:
        raise ValueError("Index cannot be None")

    if not hasattr(index, "ntotal") or index.ntotal == 0:
        raise ValueError("Index is empty (ntotal = 0)")

    external_ids = safe_get_id_mapping(index)

    if external_ids is None:
        # Provide helpful error message based on index type
        index_type = type(index).__name__
        if "IDMap" not in index_type:
            raise ValueError(
                f"Index type {index_type} does not support ID mapping. "
                f"Use IndexIDMap or IndexIDMap2 for external ID support."
            )
        else:
            raise RuntimeError(
                f"Failed to access ID mapping from {index_type}. "
                f"This may indicate a FAISS version compatibility issue."
            )

    if len(external_ids) != index.ntotal:
        logger.warning(f"ID mapping size ({len(external_ids)}) doesn't match index size ({index.ntotal})")

    return external_ids


def build_id_lookup(index: Any) -> Dict[int, int]:
    """
    Build a lookup dictionary from external ID to internal index position.

    This function creates a mapping that allows efficient lookup of internal
    FAISS positions based on external document IDs.

    Parameters
    ----------
    index : Any
        A FAISS index object with ID mapping capabilities.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping external ID to internal index position.

    Raises
    ------
    ValueError
        If the index doesn't support ID mapping.

    Examples
    --------
    >>> id_lookup = build_id_lookup(index)
    >>> internal_pos = id_lookup.get(external_id)
    >>> if internal_pos is not None:
    ...     # Use internal_pos for FAISS operations
    ...     pass
    """
    external_ids = get_faiss_external_ids(index)
    return {external_id: internal_idx for internal_idx, external_id in enumerate(external_ids)}


def check_faiss_index_compatibility(index: Any) -> Dict[str, Any]:
    """
    Check FAISS index capabilities and return compatibility information.

    This function inspects a FAISS index to determine what operations
    are supported and provides debugging information.

    Parameters
    ----------
    index : Any
        A FAISS index object to inspect.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing compatibility information:
        - 'has_id_mapping': bool, whether ID mapping is available
        - 'index_type': str, the index type name
        - 'ntotal': int, number of vectors in the index
        - 'supports_reconstruct': bool, whether reconstruction is supported
        - 'attributes': List[str], available attributes

    Examples
    --------
    >>> info = check_faiss_index_compatibility(index)
    >>> if info['has_id_mapping']:
    ...     print("Index supports external ID mapping")
    >>> print(f"Index type: {info['index_type']}")
    """
    if index is None:
        return {
            "has_id_mapping": False,
            "index_type": "None",
            "ntotal": 0,
            "supports_reconstruct": False,
            "attributes": [],
        }

    index_type = type(index).__name__
    attributes = [attr for attr in dir(index) if not attr.startswith("_")]

    # Check for ID mapping support
    has_id_mapping = safe_get_id_mapping(index) is not None

    # Check for reconstruction support
    supports_reconstruct = hasattr(index, "reconstruct") and callable(getattr(index, "reconstruct", None))

    return {
        "has_id_mapping": has_id_mapping,
        "index_type": index_type,
        "ntotal": getattr(index, "ntotal", 0),
        "supports_reconstruct": supports_reconstruct,
        "attributes": attributes,
    }
