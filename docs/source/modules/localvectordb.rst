localvectordb package
=====================

Everything below is re-exported at the top level: import it as
``from localvectordb import LocalVectorDB``, not from the submodule it happens
to be defined in.

This page renders the whole public surface for browsing, but it deliberately
does **not** own the cross-reference targets -- each symbol is documented
canonically once, on the page for the module that defines it (linked under
Submodules below). Documenting it in both places is what produced Sphinx's
"more than one target found" warnings, since autodoc emits unqualified
references from type annotations and cannot choose between two equal targets.
Hence ``:no-index:``; removing it re-introduces ~75 warnings.

.. automodule:: localvectordb
   :members:
   :show-inheritance:
   :undoc-members:
   :no-index:

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   localvectordb.database
   localvectordb.extractors
   localvectordb.validation
   localvectordb.visualization

Submodules
----------

.. toctree::
   :maxdepth: 4

   localvectordb.backup
   localvectordb.chunking
   localvectordb.client
   localvectordb.core
   localvectordb.cursor
   localvectordb.embeddings
   localvectordb.exceptions
   localvectordb.factory
   localvectordb.migration
   localvectordb.patching
   localvectordb.query_builder
   localvectordb.reranking
   localvectordb.section_detection
   localvectordb.section_metadata
   localvectordb.sqlite_tuning
   localvectordb.utils
   localvectordb.versioning
