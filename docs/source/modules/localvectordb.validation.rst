localvectordb.validation package
=================================

The fact-checking ("reverse RAG") module — see :doc:`/factcheck` for the guide.

:class:`~localvectordb.FactChecker`, :class:`~localvectordb.FactCheckResult`,
:class:`~localvectordb.ClaimResult` and :class:`~localvectordb.Polarity` are
re-exported at the top level and documented under :doc:`localvectordb`. The LLM
provider adapters are only importable from this package, and are below.

Submodules
----------

.. toctree::
   :maxdepth: 4

   localvectordb.validation.llm
