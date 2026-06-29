---
name: fact-checking
description: Verify text claims against a knowledge base using LocalVectorDB's FactChecker. Use when the user wants to fact-check LLM output, validate statements against source documents, detect contradictions, or ground generated text in trusted references.
license: MIT
compatibility: Requires Python 3.12+, localvectordb, and an LLM client (Anthropic, OpenAI, or Google).
metadata:
  author: localvectordb
  version: "1.0"
---

# Fact-Checking with LocalVectorDB

The `FactChecker` module validates text claims against documents stored in a LocalVectorDB instance. It extracts claims from input text, searches for supporting or contradicting evidence, and produces a grounded confidence score.

## How It Works

1. **Claim extraction** - An LLM breaks the input text into individual factual claims
2. **Evidence retrieval** - Each claim is searched against the vector database using hybrid search
3. **Polarity classification** - The LLM classifies each claim-evidence pair as SUPPORTS, CONTRADICTS, or UNRELATED
4. **Scoring** - Results are aggregated into an overall grounding score with citation text

## Quick Start

```python
from anthropic import Anthropic
from localvectordb import LocalVectorDB
from localvectordb.validation import FactChecker

# 1. Create and populate a knowledge base
db = LocalVectorDB(
    name="knowledge_base",
    base_path="./kb",
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
)

db.upsert(
    documents=[
        "The company's annual leave policy grants 15 days of paid time off per year.",
        "Remote work is permitted up to 3 days per week with manager approval.",
        "The Q3 2024 revenue was $4.2 million, a 12% increase year-over-year.",
    ],
    ids=["policy-pto", "policy-remote", "financials-q3"],
)

# 2. Create a FactChecker with an LLM client
client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
checker = FactChecker(databases=db, llm=client)

# 3. Check text for accuracy
result = checker.check(
    "Employees receive 10 days of PTO and can work remotely 5 days a week."
)

print(f"Overall score: {result.overall_score:.2f}")
print(f"Has contradictions: {result.has_contradictions}")
print(f"\nCitations:\n{result.citation_text}")
```

## FactChecker Parameters

```python
checker = FactChecker(
    databases=db,                  # LocalVectorDB or list of databases
    llm=client,                    # Anthropic, OpenAI, or Google client
    model=None,                    # Optional model override
    similarity_threshold=0.3,      # Min similarity to consider a chunk relevant
    min_grounding_score=0.7,       # Min polarity confidence to count as grounded
    search_type="hybrid",          # "vector", "keyword", or "hybrid"
    top_k=5,                       # Chunks retrieved per claim per database
    max_concurrent=5,              # Max concurrent claim processing
)
```

### Supported LLM clients

| Client | Default model |
|--------|---------------|
| `anthropic.Anthropic()` | claude-3-5-haiku |
| `openai.OpenAI()` | gpt-4o-mini |
| `google.generativeai` | gemini-2.0-flash |

The `model` parameter overrides the default. You can also implement a custom `LLMProvider` class.

## Working with Results

### FactCheckResult

```python
result = checker.check("Some text to verify.")

result.claims              # List[ClaimResult] - individual claim verdicts
result.overall_score       # float (0-1) - average confidence; 0.0 if contradictions
result.has_contradictions  # bool - True if any claim is contradicted
result.citation_text       # str - formatted source references
result.annotated_text      # Optional[str] - original text with inline citations
```

### ClaimResult

```python
for claim in result.claims:
    claim.claim              # str - the extracted claim
    claim.grounded           # bool - whether evidence supports it
    claim.confidence         # float (0-1) - polarity confidence
    claim.polarity           # Polarity.SUPPORTS, .CONTRADICTS, or .UNRELATED
    claim.source_id          # str - document ID of best evidence
    claim.source_excerpt     # str - best matching quote from source
    claim.contradiction      # bool - True if contradicted
    claim.similarity         # float - vector similarity score
    claim.database_name      # str - which database matched
```

### Polarity enum

```python
from localvectordb.validation import Polarity

Polarity.SUPPORTS     # Source supports the claim
Polarity.CONTRADICTS  # Source contradicts the claim
Polarity.UNRELATED    # Source doesn't relate to claim
```

## Multi-Database Fact-Checking

Check claims against multiple knowledge bases simultaneously:

```python
policy_db = LocalVectorDB(name="policies", base_path="./dbs")
financial_db = LocalVectorDB(name="financials", base_path="./dbs")

checker = FactChecker(
    databases=[policy_db, financial_db],
    llm=client,
)

# Claims are searched across all databases
result = checker.check("The company had $5M revenue and offers 20 PTO days.")
```

## Scoped Checking

Restrict verification to specific source documents:

```python
result = checker.check(
    "Remote work is allowed every day.",
    sources=["policy-remote"],  # Only check against these document IDs
)
```

## Common Patterns

### Validate LLM output before presenting to users

```python
import anthropic

client = anthropic.Anthropic()

# Generate a response
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize the Q3 financials."}],
)
generated_text = response.content[0].text

# Fact-check the response against your knowledge base
checker = FactChecker(databases=db, llm=client)
result = checker.check(generated_text)

if result.has_contradictions:
    print("WARNING: Generated text contains contradictions!")
    for claim in result.claims:
        if claim.contradiction:
            print(f"  Contradicted: {claim.claim}")
            print(f"  Source says: {claim.source_excerpt}")
elif result.overall_score < 0.5:
    print("WARNING: Low grounding score - claims may not be supported")
else:
    print(f"Verified (score: {result.overall_score:.2f})")
    print(generated_text)
```

### Build a grounded Q&A system

```python
def grounded_answer(question: str, db, llm_client) -> str:
    # Search for relevant context
    results = db.query(question, search_type="hybrid", k=5)
    context = "\n".join(r.content for r in results)

    # Generate answer with context
    response = llm_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system="Answer based only on the provided context.",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    answer = response.content[0].text

    # Verify the answer
    checker = FactChecker(databases=db, llm=llm_client)
    check = checker.check(answer)

    return f"{answer}\n\n---\nGrounding score: {check.overall_score:.2f}\n{check.citation_text}"
```

## Tuning Tips

- **Lower `similarity_threshold`** (e.g. 0.2) to cast a wider net for evidence retrieval
- **Raise `min_grounding_score`** (e.g. 0.8) for stricter verification
- **Increase `top_k`** if documents are long and claims span multiple chunks
- **Use `search_type="hybrid"`** for best recall (combines semantic + keyword matching)
- **Use `sources`** parameter to scope checking when you know which documents are relevant
