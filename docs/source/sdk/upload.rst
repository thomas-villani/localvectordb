File Upload
===========

The SDK supports uploading files to a database with automatic server-side text extraction.
Supported formats include PDF, DOCX, PPTX, XLSX, TXT, Markdown, and more.

Extraction is powered by `all2md <https://all2md.readthedocs.io/>`_ and returns
**Markdown**, preserving headings, tables, and lists — which makes the server's
structure-aware chunking (e.g. ``sections``) more effective. If you run your own
server, the extended-format parsers come from the ``file-extraction`` extra
(and ``file-extraction-ocr`` for scanned PDFs). See :doc:`/file-extraction`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Browser
-------

In the browser, pass native ``File`` objects from a file input or drag-and-drop:

.. code-block:: typescript

   const input = document.querySelector<HTMLInputElement>("#file-input")!;

   input.addEventListener("change", async () => {
     const files = Array.from(input.files ?? []);

     const result = await db.upload(files, {
       metadata: { source: "web-upload" },
     });

     console.log(`Uploaded ${result.files_processed} files`);
     console.log("Document IDs:", result.document_ids);
   });

Node.js
-------

In Node.js, read files into a ``Buffer`` or ``Uint8Array`` and pass them as objects with a
``name``, ``data``, and optional ``type``:

.. code-block:: typescript

   import { readFile } from "fs/promises";

   const pdf = await readFile("report.pdf");
   const docx = await readFile("notes.docx");

   const result = await db.upload([
     { name: "report.pdf", data: pdf, type: "application/pdf" },
     { name: "notes.docx", data: docx, type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" },
   ]);

.. tip::

   On Node.js 20+, the global ``File`` constructor is available, so you can also do:

   .. code-block:: typescript

      const file = new File([pdf], "report.pdf", { type: "application/pdf" });
      await db.upload([file]);

Upload Options
--------------

.. code-block:: typescript

   await db.upload(files, {
     metadata: { department: "research" },  // Applied to all uploaded documents
     batch_size: 50,                        // Documents per batch (default: 100)
     ids: ["custom-id-1", "custom-id-2"],   // Custom document IDs
     mode: "upsert",                        // "upsert" (default) or "insert"
     errors: "raise",                       // "raise" (default) or "ignore"
     similarity_threshold: 0.95,            // Skip near-duplicate chunks
     use_filename_as_id: true,              // Use the filename as the document ID
   });

UploadableFile Type
-------------------

The ``upload()`` method accepts an array of ``UploadableFile``, which is a union type:

.. code-block:: typescript

   type UploadableFile =
     | File                                              // Browser File object
     | Blob                                              // Browser or Node.js Blob
     | { name: string; data: Blob | ArrayBuffer | Uint8Array; type?: string };  // Node.js convenience

This allows the same method to work across all runtimes without requiring any Node.js-specific
imports (like ``fs``) in the SDK itself.
