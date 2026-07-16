.. _file-extraction:

=============================
File Extraction System
=============================

LocalVectorDB extracts text from document files so you can upload them directly
to your vector database without preparing the text yourself. Extraction is
powered by the `all2md <https://all2md.readthedocs.io/>`_ library, which converts
20+ document formats and 200+ source/text formats to Markdown.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Extraction is exposed through a small **plugin architecture** (Python entry
points). A single built-in extractor, :class:`~localvectordb.extractors.all2md_extractor.All2MdExtractor`,
delegates to all2md and covers every built-in format. The plugin interface
remains so you can register your own extractor for a format all2md does not
handle, or to override the default behaviour for a specific format.

Key characteristics
^^^^^^^^^^^^^^^^^^^^

- **Markdown output**: extracted content is Markdown, preserving structure
  (headings, tables, lists) that downstream chunking and section detection can
  use for better boundaries.
- **Dependency-aware**: the formats reported as supported reflect which of
  all2md's optional parser dependencies are actually installed.
- **Hardened by default**: untrusted uploads are converted with remote fetching
  and local-file access disabled, and (for HTML) dangerous elements stripped.
- **Extensible**: register additional extractors via the
  ``localvectordb.file_extractors`` entry-point group.

Architecture components
^^^^^^^^^^^^^^^^^^^^^^^^

**ExtractorRegistry**
  Central registry that discovers and selects extractors.

**BaseExtractor**
  Abstract base class all extractors inherit from.

**All2MdExtractor**
  The built-in extractor that delegates to all2md.

**ExtractionResult**
  Standardized result object containing extracted text (Markdown), metadata, and
  status.

Supported file formats
----------------------

The common document formats work out of the box with a base installation,
because all2md (and the extras needed for these formats) is a core dependency:

- **Documents**: PDF, Word (``.docx``), PowerPoint (``.pptx``), Excel
  (``.xlsx``), HTML/MHTML, EPUB, RTF, OpenDocument (``.odt``/``.odp``/``.ods``)
- **Markup / data**: Markdown, reStructuredText, Org-Mode, OpenAPI/Swagger,
  CSV/TSV, JSON, YAML, TOML, INI, Jupyter notebooks (``.ipynb``)
- **Email**: ``.eml``
- **Source code and plain text**: 200+ extensions

Extended and less-common formats (LaTeX, MediaWiki/wiki, Textile, archives,
Evernote ``.enex``, FictionBook ``.fb2``, Outlook) are available via the
``file-extraction`` extra. Optical character recognition for scanned PDFs is
available via the ``file-extraction-ocr`` extra.

To see exactly which formats are available in your environment:

.. code-block:: python

   from localvectordb.extractors import get_supported_formats

   formats = get_supported_formats()
   for name, info in sorted(formats.items()):
       print(name, info["extensions"])

Installation
------------

.. code-block:: bash

   # Common document formats work with the base install
   # (uv recommended; swap `uv add` for `pip install` if you prefer pip)
   uv add localvectordb

   # Extended / niche formats (latex, wiki, textile, archives, ...)
   uv add "localvectordb[file-extraction]"

   # OCR for scanned/image-only PDFs (also requires the Tesseract system binary)
   uv add "localvectordb[file-extraction-ocr]"

   # Everything
   uv add "localvectordb[all]"

.. note::

   The ``pdf-layout`` and EasyOCR all2md extras are intentionally **not**
   bundled: ``pdf-layout`` is distributed under a noncommercial license that is
   incompatible with this project's MIT license, and EasyOCR pulls in a heavy
   PyTorch dependency. Install them yourself (``pip install all2md[pdf-layout]``
   / ``all2md[ocr-easyocr]``) only if their licensing/footprint is acceptable
   for your use case.

Using the extraction system
----------------------------

Server upload API
^^^^^^^^^^^^^^^^^

The most common way to use file extraction is through the server upload API:

.. code-block:: bash

   curl -X POST \
        -H "Authorization: Bearer your_api_key" \
        -F "files=@document.pdf" \
        -F "metadata={\"category\": \"research\"}" \
        http://localhost:8000/api/v1/databases/mydatabase/upload

Direct extraction
^^^^^^^^^^^^^^^^^

You can use the extraction system directly without uploading:

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   with open("document.pdf", "rb") as f:
       result = ExtractorRegistry.extract_text(file_content=f.read(), filename="document.pdf")

   if result.success:
       print(result.text[:500])          # Markdown
       print(result.method)              # e.g. "All2MdExtractor:pdf"
       print(result.metadata)            # title, author, source_format, ...
   else:
       print("Extraction failed:", result.error)

Ingesting files into a database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~localvectordb.database.LocalVectorDB` can ingest files directly, running them
through the extractor before chunking and embedding:

.. code-block:: python

   from localvectordb import LocalVectorDB

   db = LocalVectorDB("documents")
   db.upsert_from_file(["report.docx", "notes.md", "paper.pdf"])

Security
--------

When converting untrusted uploads, the extractor applies hardened defaults:

- remote asset fetching is disabled (no SSRF surface),
- remote document fetching is disabled,
- local ``file://`` access is disabled,
- HTML scripts and event handlers are stripped, and
- embedded attachments are skipped.

A base file-size guard and a ZIP-bomb guard (for ZIP-based formats such as
``.docx``/``.xlsx``/``.pptx``/``.epub``/``.odt``) run before content is handed to
all2md.

These defaults can be relaxed for trusted content through the server's
``[extraction]`` configuration section:

.. list-table:: ``[extraction]`` settings
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Description
   * - ``allow_remote_fetch``
     - ``false``
     - Allow fetching remote assets referenced by a document.
   * - ``allowed_hosts``
     - ``None``
     - Host allowlist applied when ``allow_remote_fetch`` is enabled.
   * - ``strip_dangerous_elements``
     - ``true``
     - HTML only: strip scripts / event handlers.
   * - ``attachment_mode``
     - ``"skip"``
     - How embedded attachments/assets are handled.

.. code-block:: toml

   # config.toml
   [extraction]
   allow_remote_fetch = false
   strip_dangerous_elements = true
   attachment_mode = "skip"

The same settings can be supplied via environment variables, e.g.
``LVDB_EXTRACTION_ALLOW_REMOTE_FETCH=true``.

Metadata extraction
-------------------

all2md returns document metadata (such as ``title``, ``author`` and
``language``) which the extractor merges with a few standard fields
(``filename``, ``source_format``, ``file_size_bytes``, ``character_count``).
Only metadata keys that exist in the target database's metadata schema are
persisted; unknown keys are ignored.

.. code-block:: python

   db = LocalVectorDB(
       name="documents",
       metadata_schema={
           "title": {"type": "text", "indexed": True},
           "author": {"type": "text", "indexed": True},
           "source_format": {"type": "text", "indexed": True},
       },
   )
   db.upsert_from_file(["research_paper.pdf"])
   results = db.filter(where={"author": "Jane Smith"})

Extractor priority and selection
--------------------------------

When several extractors can handle the same file, the registry selects the
highest-priority one (priority only matters among extractors that claim the same
format). The built-in ``All2MdExtractor`` uses priority ``10``, so a custom
extractor registered with a higher priority will take precedence for the formats
it claims, while all2md remains the default for everything else.

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   extractors = ExtractorRegistry.get_extractors_for_file("document.pdf")
   for extractor in extractors:
       print(extractor.name, extractor.priority)

Creating custom extractors
---------------------------

You can extend the system with custom extractors for specialized formats or to
override the default behaviour.

.. code-block:: python

   from localvectordb.extractors import BaseExtractor, ExtractionResult
   from localvectordb.core import MetadataField

   class CustomFormatExtractor(BaseExtractor):
       @property
       def supported_extensions(self):
           return [".myfmt"]

       @property
       def supported_mimetypes(self):
           return ["application/x-myfmt"]

       @property
       def required_packages(self):
           return []

       @property
       def priority(self):
           return 20  # higher than All2MdExtractor (10) -> wins for .myfmt

       @property
       def metadata_schema(self):
           return {"records": MetadataField(type="integer", indexed=True)}

       def _check_availability(self):
           return True

       def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
           text = file_content.decode("utf-8", errors="ignore")
           return ExtractionResult(text=text, success=True, method=self.name, metadata={})

Register it directly:

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   ExtractorRegistry.register(CustomFormatExtractor)

or, for a distributable package, via an entry point in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."localvectordb.file_extractors"]
   myfmt = "mypackage.extractors:CustomFormatExtractor"

The extractor is discovered automatically when LocalVectorDB starts.

Troubleshooting
---------------

**"Missing optional dependency for '<format>'"**
  Install the all2md extra that provides the parser, e.g.
  ``pip install "localvectordb[file-extraction]"`` for extended formats or
  ``pip install "localvectordb[file-extraction-ocr]"`` for scanned PDFs.

**"Unsupported or undetectable format"**
  The file's extension is not recognised and its content could not be detected.
  Confirm the extension matches the content, or register a custom extractor.

**Enable debug logging**

.. code-block:: python

   import logging
   logging.getLogger("localvectordb.extractors").setLevel(logging.DEBUG)

See Also
--------

- :doc:`/installation` - Installing optional dependencies
- :doc:`/cli` - Command-line file upload tools
- `all2md documentation <https://all2md.readthedocs.io/>`_
