Fact-Checking (Reverse RAG)
===========================

The ``localvectordb.validation`` module verifies LLM-generated text against the documents
you have already stored. Where ordinary RAG retrieves context *before* generating an answer,
fact-checking runs *after*: it breaks a piece of text into individual factual claims, searches
your databases for evidence, and uses an LLM to decide whether each claim is **supported**,
**contradicted**, or **unrelated** to the retrieved sources. This "reverse RAG" pass is a
practical way to catch hallucinations before showing model output to a user.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

The pipeline has three stages, all orchestrated by :class:`FactChecker`:

1. **Claim extraction** -- the LLM splits the input text into atomic factual claims.
2. **Retrieval** -- each claim is used as a query against one or more
   :class:`~localvectordb.LocalVectorDB` instances (vector, keyword, or hybrid search).
3. **Polarity classification** -- the LLM judges each retrieved chunk against the claim and
   assigns a :class:`Polarity` (``SUPPORTS``, ``CONTRADICTS``, or ``UNRELATED``) with a
   confidence score.

All of the public classes are re-exported from the top-level package:

.. code-block:: python

   from localvectordb import FactChecker, FactCheckResult, ClaimResult, Polarity

The provider adapters live one level down, in :mod:`localvectordb.validation`:

.. code-block:: python

   from localvectordb.validation import (
       AnthropicProvider,
       OpenAIProvider,
       GeminiProvider,
       LLMProvider,
   )

Basic Usage
-----------

Construct a :class:`FactChecker` with one or more databases and an LLM client, then call
:meth:`~FactChecker.check`:

.. code-block:: python

   import anthropic
   from localvectordb import LocalVectorDB, FactChecker

   db = LocalVectorDB("knowledge.db")
   client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

   checker = FactChecker(databases=db, llm=client)

   result = checker.check("The policy allows 10 days of PTO per year.")

   print(f"Overall score: {result.overall_score:.2f}")
   print(f"Contradictions: {result.has_contradictions}")

``databases`` accepts either a single ``LocalVectorDB`` or a list of them. ``llm`` accepts an
Anthropic, OpenAI, or Google GenAI client (the provider is auto-detected), or any object
implementing the :class:`LLMProvider` protocol.

The :class:`FactChecker` constructor signature is:

.. code-block:: python

   FactChecker(
       databases,                     # LocalVectorDB | list[LocalVectorDB]
       llm,                           # an LLM client or LLMProvider
       model=None,                    # provider-specific default when None
       similarity_threshold=0.3,      # min retrieval score for a chunk to count
       min_grounding_score=0.7,       # min polarity confidence to call a claim grounded
       search_type="hybrid",          # "vector", "keyword", or "hybrid"
       top_k=5,                       # chunks retrieved per claim per database
       max_concurrent=5,              # claims processed concurrently
   )

Async
^^^^^

:meth:`~FactChecker.check` is a thin synchronous wrapper around
:meth:`~FactChecker.check_async`. In an async application, call the coroutine directly to
avoid nesting event loops:

.. code-block:: python

   result = await checker.check_async(text)

Both methods share the same signature -- ``check(text, sources=None)`` and
``check_async(text, sources=None)`` -- and return a :class:`FactCheckResult`.

Interpreting the Result
-----------------------

:meth:`~FactChecker.check` returns a :class:`FactCheckResult` with the following fields:

- ``claims`` -- a list of :class:`ClaimResult` objects, one per extracted claim.
- ``overall_score`` -- a float in ``[0.0, 1.0]``. It is the mean claim confidence, but is
  forced to ``0.0`` if *any* claim is contradicted, and is ``1.0`` when no factual claims are
  detected.
- ``has_contradictions`` -- ``True`` if any claim was contradicted by a source.
- ``citation_text`` -- a human-readable summary of sources consulted, contradictions, and
  ungrounded claims.
- ``annotated_text`` -- the original text with inline ``[N]`` citation markers and appended
  footnotes for grounded claims, or ``None`` if no annotations could be placed.

Each :class:`ClaimResult` carries:

- ``claim`` -- the extracted factual claim.
- ``grounded`` -- ``True`` if a supporting source was found above ``min_grounding_score``.
- ``confidence`` -- polarity confidence for the best supporting chunk (``0.0`` when
  contradicted or ungrounded).
- ``source_id`` -- the document ID of the matched source (or ``None``).
- ``source_excerpt`` -- the excerpt the LLM cited as evidence.
- ``contradiction`` -- ``True`` if the claim was contradicted.
- ``polarity`` -- a :class:`Polarity` value (``SUPPORTS``, ``CONTRADICTS``, ``UNRELATED``).
- ``similarity`` -- the retrieval score of the matched chunk.
- ``original_sentence`` -- the sentence in the input text the claim came from.
- ``database_name`` -- the name of the database the source was found in.

.. code-block:: python

   result = checker.check(answer_text)

   for claim in result.claims:
       status = (
           "CONTRADICTED" if claim.contradiction
           else "grounded" if claim.grounded
           else "ungrounded"
       )
       print(f"[{status}] {claim.claim}")
       if claim.source_id:
           print(f"    source: {claim.source_id} ({claim.database_name})")
           print(f"    excerpt: {claim.source_excerpt}")

   # Human-readable summary of sources, contradictions, and ungrounded claims
   print(result.citation_text)

   # Inline-cited version of the original text (may be None)
   if result.annotated_text:
       print(result.annotated_text)

The ``Polarity`` enum is a ``str`` enum, so its members compare and serialise as plain
strings:

.. code-block:: python

   from localvectordb import Polarity

   assert Polarity.SUPPORTS == "supports"
   assert Polarity.CONTRADICTS.value == "contradicts"

Scoping to Source Documents
---------------------------

If you already know which documents were used to generate the text (for example, the chunks
your RAG pipeline retrieved), pass their document IDs via ``sources``. This enables a
two-phase search:

1. **Scoped phase** -- only the listed source documents are searched. If a claim is supported
   here above ``min_grounding_score``, the check short-circuits and returns immediately.
2. **Expanded phase** -- if the scoped search finds no support (or the claim is unsupported),
   the checker falls back to searching *all* documents across *all* databases. This catches
   claims that are unsupported by, or even contradicted by, the wider knowledge base.

.. code-block:: python

   # These were the documents the LLM was given as context
   context_doc_ids = ["policy-2026", "handbook-benefits"]

   result = checker.check(
       "The policy allows 10 days of PTO per year.",
       sources=context_doc_ids,
   )

This is the recommended pattern for validating a specific RAG response: it verifies the answer
against its own cited context first, then widens the net only when that context is insufficient.

Tuning Thresholds
-----------------

Two thresholds control sensitivity:

- ``similarity_threshold`` (default ``0.3``) is the minimum *retrieval* score for a chunk to be
  considered at all. Raise it to only classify high-relevance chunks; lower it to surface more
  candidate evidence.
- ``min_grounding_score`` (default ``0.7``) is the minimum *polarity confidence* for a claim to
  be marked ``grounded``. Raise it for stricter grounding; lower it to be more permissive.

``search_type`` and ``top_k`` control retrieval breadth, and ``max_concurrent`` bounds how many
claims are classified in parallel:

.. code-block:: python

   checker = FactChecker(
       databases=[db_docs, db_wiki],
       llm=client,
       similarity_threshold=0.4,   # stricter retrieval
       min_grounding_score=0.8,    # stricter grounding
       search_type="vector",       # pure semantic search
       top_k=8,                     # more evidence per claim
       max_concurrent=10,           # more parallelism
   )

Selecting and Customising the LLM Provider
------------------------------------------

The provider is auto-detected from the client object's module, so passing a native SDK client
is usually all you need:

.. code-block:: python

   # Anthropic
   import anthropic
   checker = FactChecker(databases=db, llm=anthropic.Anthropic())

   # OpenAI
   import openai
   checker = FactChecker(databases=db, llm=openai.OpenAI())

   # Google GenAI
   from google import genai
   checker = FactChecker(databases=db, llm=genai.Client())

Both synchronous and asynchronous SDK clients are supported (for example
``anthropic.AsyncAnthropic`` or ``openai.AsyncOpenAI``); the adapter detects which flavour it
was given.

Override the model with the ``model`` parameter. When left as ``None``, each provider uses its
own default (``claude-haiku-4-5-20251001`` for Anthropic, ``gpt-4o-mini`` for OpenAI,
``gemini-2.0-flash`` for Gemini):

.. code-block:: python

   checker = FactChecker(
       databases=db,
       llm=anthropic.Anthropic(),
       model="claude-sonnet-4-5",
   )

You can also construct a provider adapter explicitly:

.. code-block:: python

   from localvectordb.validation import AnthropicProvider

   provider = AnthropicProvider(anthropic.Anthropic(), model="claude-sonnet-4-5")
   checker = FactChecker(databases=db, llm=provider)

Custom Providers
^^^^^^^^^^^^^^^^

To use any other LLM, implement the :class:`LLMProvider` protocol -- a single async
``complete`` method that takes a system prompt and a user prompt and returns the model's text
response:

.. code-block:: python

   class MyProvider:
       async def complete(self, system: str, user: str) -> str:
           # call your model however you like and return its text
           ...

   checker = FactChecker(databases=db, llm=MyProvider())

Because :class:`LLMProvider` is a runtime-checkable ``Protocol``, any object with a matching
``complete`` signature is accepted without subclassing.

HTTP Server
-----------

The same functionality is exposed over HTTP when running the LocalVectorDB server. See the
fact-checking endpoints in :doc:`server/routes` (``POST /api/v1/{db_name}/factcheck`` for a
single database and ``POST /api/v1/factcheck`` across databases). Those endpoints require the
``localvectordb`` validation module to be installed and accept ``llm_provider``,
``llm_api_key``, ``model``, ``similarity_threshold``, ``min_grounding_score``, ``search_type``,
and ``k`` in the request body.
