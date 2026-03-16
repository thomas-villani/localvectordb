Agent Skills
============

LocalVectorDB ships with pre-built `Agent Skills <https://agentskills.io>`_ -- portable
instruction packages that teach AI coding agents how to use LocalVectorDB features
effectively. Skills are supported by Claude Code, Cursor, VS Code Copilot, Gemini CLI,
and many other agent products.

.. contents:: On This Page
   :local:
   :depth: 2

What Are Skills?
----------------

Agent Skills are folders containing a ``SKILL.md`` file with structured instructions that
an agent loads on demand when a relevant task is detected. They follow the open
`Agent Skills specification <https://agentskills.io/specification>`_ and work across any
compatible agent product.

Each skill has:

- **Metadata** (name, description) -- loaded at startup for task matching
- **Instructions** -- loaded when the skill is activated
- **Optional resources** -- scripts, references, assets loaded as needed

Available Skills
----------------

LocalVectorDB includes three skills in the ``skills/`` directory:

``semantic-search``
^^^^^^^^^^^^^^^^^^^

**When it activates:** The user wants to create a vector database, add documents, or run
semantic, keyword, or hybrid searches.

**What it covers:**

- Creating databases with typed metadata schemas
- Document ingestion (``upsert``, ``insert``, batch, deduplication)
- Vector, keyword, and hybrid search via ``query()``
- Metadata filtering with ``filter()``
- CRUD operations (``get``, ``update``, ``delete``, ``count``)
- Embedding provider and chunking strategy configuration
- The factory pattern for local/remote databases

``fact-checking``
^^^^^^^^^^^^^^^^^

**When it activates:** The user wants to verify LLM-generated text against a knowledge base,
detect contradictions, or build grounded Q&A systems.

**What it covers:**

- Setting up a ``FactChecker`` with a database and LLM client
- Checking text for accuracy and grounding
- Working with ``FactCheckResult``, ``ClaimResult``, and ``Polarity``
- Multi-database fact-checking
- Scoped verification against specific source documents
- Patterns for validating LLM output before presenting to users

``document-comparison``
^^^^^^^^^^^^^^^^^^^^^^^

**When it activates:** The user wants to measure document similarity, find nearest neighbours,
detect partial overlap, cluster documents, or create embedding visualisations.

**What it covers:**

- Pairwise document comparison (``compare_documents``)
- Nearest-neighbour search (``nearest_neighbors``)
- Similarity matrices (``pairwise_similarity_matrix``)
- Chunk-level detailed comparison (``compare_documents_detailed``)
- Embedding maps, heatmaps, cluster plots, and similarity graphs
- Interactive plotly visualisations

Using Skills
------------

With Claude Code
^^^^^^^^^^^^^^^^

If the skills are in a repository that Claude Code has access to, they are discovered
automatically. Claude will activate the relevant skill when your task matches its
description.

You can also install them explicitly:

.. code-block:: bash

   # From the anthropic skills marketplace (if published)
   /install skills/semantic-search

With Other Agents
^^^^^^^^^^^^^^^^^

Most skills-compatible agents discover ``SKILL.md`` files in the repository automatically.
Check your agent's documentation for specifics:

- `Cursor <https://cursor.com/docs/context/skills>`_
- `VS Code Copilot <https://code.visualstudio.com/docs/copilot/customization/agent-skills>`_
- `Gemini CLI <https://geminicli.com/docs/cli/skills/>`_
- `Roo Code <https://docs.roocode.com/features/skills>`_

See the full list of supported agents at `agentskills.io <https://agentskills.io>`_.

Creating Custom Skills
----------------------

You can create your own skills for project-specific LocalVectorDB workflows. A skill is a
directory with a ``SKILL.md`` file:

.. code-block:: text

   my-custom-skill/
   +-- SKILL.md
   +-- scripts/          # optional
   +-- references/       # optional

The ``SKILL.md`` file requires YAML frontmatter with ``name`` and ``description``:

.. code-block:: markdown

   ---
   name: my-custom-skill
   description: Describe what this skill does and when to use it. Include keywords
     that help agents identify relevant tasks.
   ---

   # My Custom Skill

   Instructions go here. Write clear, step-by-step guidance with code examples.

**Naming rules:**

- Lowercase letters, numbers, and hyphens only
- 1-64 characters
- Must not start or end with a hyphen
- No consecutive hyphens
- Must match the parent directory name

**Best practices:**

- Keep ``SKILL.md`` under 500 lines; move reference material to separate files
- Write a descriptive ``description`` with specific keywords for task matching
- Include complete, runnable code examples
- Cover common edge cases and error handling
- Test the skill by asking an agent to perform tasks that should trigger it

Skill Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   skills/
   +-- semantic-search/
   |   +-- SKILL.md
   +-- fact-checking/
   |   +-- SKILL.md
   +-- document-comparison/
   |   +-- SKILL.md
   +-- my-custom-skill/
       +-- SKILL.md
       +-- scripts/
       |   +-- setup.py
       +-- references/
           +-- api-reference.md

Skills placed in the ``skills/`` directory at the repository root will be discovered
by compatible agents automatically.
