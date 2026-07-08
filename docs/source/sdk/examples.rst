Examples
========

Real-world usage patterns for the JavaScript / TypeScript SDK.

.. contents:: Table of Contents
   :local:
   :depth: 2

RAG (Retrieval-Augmented Generation)
-------------------------------------

Use LocalVectorDB as the retrieval layer for a RAG pipeline:

.. code-block:: typescript

   import { LocalVectorDBClient } from "@localvectordb/sdk";
   import Anthropic from "@anthropic-ai/sdk";

   const lvdb = new LocalVectorDBClient({ baseUrl: "http://localhost:8000" });
   const db = lvdb.database("knowledge_base");
   const anthropic = new Anthropic();

   async function askQuestion(question: string): Promise<string> {
     // 1. Retrieve relevant documents
     const { results } = await db.query(question, {
       search_type: "hybrid",
       k: 5,
       score_threshold: 0.3,
     });

     // 2. Build context from results
     const context = results
       .map((r) => `[${r.id}] (score: ${r.score.toFixed(2)})\n${r.content}`)
       .join("\n\n---\n\n");

     // 3. Generate answer with context
     const response = await anthropic.messages.create({
       model: "claude-sonnet-4-20250514",
       max_tokens: 1024,
       messages: [
         {
           role: "user",
           content: `Based on the following documents, answer the question.

   Documents:
   ${context}

   Question: ${question}`,
         },
       ],
     });

     return response.content[0].type === "text" ? response.content[0].text : "";
   }

Document Ingestion Pipeline
----------------------------

Ingest files from a directory with metadata:

.. code-block:: typescript

   import { readFile, readdir, stat } from "fs/promises";
   import { join, extname, basename } from "path";
   import { LocalVectorDBClient } from "@localvectordb/sdk";

   const client = new LocalVectorDBClient({ baseUrl: "http://localhost:8000" });
   const db = client.database("documents");

   async function ingestDirectory(dirPath: string): Promise<void> {
     const entries = await readdir(dirPath);

     for (const entry of entries) {
       const filePath = join(dirPath, entry);
       const fileStat = await stat(filePath);

       if (!fileStat.isFile()) continue;

       const ext = extname(entry).toLowerCase();
       if (![".pdf", ".docx", ".txt", ".md"].includes(ext)) continue;

       const data = await readFile(filePath);
       await db.upload(
         [{ name: entry, data, type: mimeType(ext) }],
         {
           metadata: {
             source: dirPath,
             filename: entry,
             size_bytes: fileStat.size,
             ingested_at: new Date().toISOString(),
           },
           use_filename_as_id: true,
         }
       );

       console.log(`Ingested: ${entry}`);
     }
   }

   function mimeType(ext: string): string {
     const types: Record<string, string> = {
       ".pdf": "application/pdf",
       ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
       ".txt": "text/plain",
       ".md": "text/markdown",
     };
     return types[ext] ?? "application/octet-stream";
   }

Express.js Search API
----------------------

Expose LocalVectorDB search through an Express API:

.. code-block:: typescript

   import express from "express";
   import { LocalVectorDBClient, DatabaseNotFoundError } from "@localvectordb/sdk";

   const app = express();
   app.use(express.json());

   const lvdb = new LocalVectorDBClient({
     baseUrl: process.env.LVDB_URL ?? "http://localhost:8000",
     apiKey: process.env.LVDB_API_KEY,
   });

   app.get("/api/search/:database", async (req, res) => {
     try {
       const db = lvdb.database(req.params.database);
       const { q, k = "10", type = "hybrid" } = req.query as Record<string, string>;

       if (!q) {
         return res.status(400).json({ error: "Missing query parameter 'q'" });
       }

       const results = await db.query(q, {
         search_type: type as "vector" | "keyword" | "hybrid",
         k: parseInt(k, 10),
       });

       return res.json({
         query: q,
         total: results.total_results,
         results: results.results.map((r) => ({
           id: r.id,
           score: r.score,
           content: r.content,
           metadata: r.metadata,
         })),
       });
     } catch (err) {
       if (err instanceof DatabaseNotFoundError) {
         return res.status(404).json({ error: `Database '${req.params.database}' not found` });
       }
       throw err;
     }
   });

   app.listen(3000, () => console.log("API server running on :3000"));

Browser Search Widget
---------------------

A minimal HTML search widget using the SDK via a bundler:

.. code-block:: html

   <div id="search">
     <input type="text" id="query" placeholder="Search documents..." />
     <button id="btn">Search</button>
     <ul id="results"></ul>
   </div>

   <script type="module">
     import { LocalVectorDBClient } from "@localvectordb/sdk";

     const client = new LocalVectorDBClient({
       baseUrl: "http://localhost:8000",
     });
     const db = client.database("my_docs");

     document.getElementById("btn").addEventListener("click", async () => {
       const query = document.getElementById("query").value;
       const ul = document.getElementById("results");
       ul.innerHTML = "<li>Searching...</li>";

       try {
         const { results } = await db.query(query, {
           search_type: "hybrid",
           k: 10,
         });

         ul.innerHTML = results
           .map(
             (r) =>
               `<li><strong>${r.score.toFixed(2)}</strong> — ${r.content.slice(0, 120)}...</li>`
           )
           .join("");
       } catch (err) {
         ul.innerHTML = `<li style="color:red">Error: ${err.message}</li>`;
       }
     });
   </script>

Multi-Database Comparison
-------------------------

Compare documents across databases:

.. code-block:: typescript

   const client = new LocalVectorDBClient({ baseUrl: "http://localhost:8000" });

   // Search across all databases
   const global = await client.globalSearch("climate change impacts", {
     search_type: "hybrid",
     k: 5,
   });

   for (const [dbName, hits] of Object.entries(global.results)) {
     console.log(`\n=== ${dbName} ===`);
     for (const hit of hits) {
       console.log(`  [${hit.score.toFixed(3)}] ${hit.content.slice(0, 100)}`);
     }
   }

Streaming Progress
------------------

Show a progress indicator while streaming results:

.. code-block:: typescript

   const db = client.database("large_corpus");

   let count = 0;
   const results = [];

   for await (const result of db.queryStream("complex query", { k: 200 })) {
     results.push(result);
     count++;
     process.stdout.write(`\rReceived ${count} results...`);
   }

   console.log(`\nDone! ${results.length} total results.`);
