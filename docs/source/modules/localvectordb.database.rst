localvectordb.database package
==============================

``LocalVectorDB`` is assembled from mixins and defines no members of its own, so
``:inherited-members:`` is what makes this page document anything at all. The
argument ``ABC`` names the ancestor to stop at: members from ``ABC`` and above
(i.e. ``object``) are skipped, while everything the mixins contribute is kept.
Drop the option and the class documents zero methods -- silently, with a clean
build.

.. automodule:: localvectordb.database
   :members:
   :show-inheritance:
   :undoc-members:
   :inherited-members: ABC

Submodules
----------

.. toctree::
   :maxdepth: 4

   localvectordb.database.base
