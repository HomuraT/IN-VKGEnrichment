## Dataset Structure (VKG/OBDA, Simplified)

This dataset is designed for Hybrid VKGQA research, with samples focusing on:
- **ONTOLOGY_ONLY**: Ontology-only alignment
- **UNMAPPED_TEXT**: Unmapped text utilization (RAG→VALUES injection)
- **SUBFIELD_TOKEN**: Subfield/subfragment extraction (RAG→VALUES injection)
- **HYBRID_MIXED**: Hybrid execution (SPARQL subset + injected column fusion)

Key conventions:
- No answers (gold rows) are stored; evaluation and reproduction rely on directly executable SPARQL.
- The `sparql` field must include PREFIX, ready to copy-paste and run (endpoint configured externally).
- `references` (optional): Uses only a "reference list", no longer using injection tables. Each reference contains `content/type/purpose`, and optional `triples` (triple anchors).

### Minimal Sample Structure

Fields:
- **id**: string (unique sample identifier, e.g., `DEST-SF-0001`)
- **vkg**: string (VKG name, e.g., `DEST`/`NPD`/`EasyBgee`)
- **question**: string (natural language question)
- **sample_type**: "ONTOLOGY_ONLY" | "UNMAPPED_TEXT" | "SUBFIELD_TOKEN" | "HYBRID_MIXED"
- **sparql**: string (complete SPARQL with PREFIX, directly executable)
- **references** (optional): List[Reference]
  - Reference:
    - `content`: string (reference content body)
    - `type`: "text" | "markdown" | "sparql" | "uri"
    - `purpose`: string (how to use this content)
    - `triples` (optional): List[[S, P, O]] (triple anchors)
      - S/P uses URI or CURIE; O can be URI/CURIE or literal
      - For entity-only cases, use wildcards: `["ex:Item1", "*", "*"]`
      - When annotating via web UI, triples are entered in NT format (one per line), automatically converted to `[S,P,O]` arrays on save

### Example: Subfield Extraction (References + Triples)

```json
{
  "id": "DEST-SF-0001",
  "vkg": "DEST",
  "question": "Extract entities and codes from text and align with graph.",
  "sample_type": "SUBFIELD_TOKEN",
  "sparql": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX ex:   <http://example.org/>\n\nSELECT ?entity ?token\nWHERE {\n  VALUES (?entity ?token) {\n    (ex:Item1 \"A-12\")\n    (ex:Item2 \"B-07\")\n  }\n}",
  "references": [
    {
      "content": "Entity ex:Item1 has code A-12; entity ex:Item2 has code B-07.",
      "type": "text",
      "purpose": "Extract entity-token pairs from fragment",
      "triples": [["ex:Item1", "ex:code", "\"A-12\""], ["ex:Item2", "ex:code", "\"B-07\""]]
    }
  ]
}
```

### Example: Entity-Only (Using Wildcards)

```json
{
  "id": "DEST-SF-0002",
  "vkg": "DEST",
  "question": "Align entities by name and return paired tokens.",
  "sample_type": "SUBFIELD_TOKEN",
  "sparql": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX ex:   <http://example.org/>\n\nSELECT ?entity ?name ?token\nWHERE {\n  ?entity rdfs:label ?name .\n}",
  "references": [
    { "content": "Item 1", "type": "text", "purpose": "Candidate entity name", "triples": [["*","rdfs:label","\"Item 1\""]] },
    { "content": "Item 2", "type": "text", "purpose": "Candidate entity name", "triples": [["*","rdfs:label","\"Item 2\""]] }
  ]
}
```

### Example: No Injection Required (Pure Ontology/Mapping Query)

When the query can be fully covered by VKG ontology/mappings, provide directly executable SPARQL; omit `injection_values`.

```json
{
  "id": "DEST-ONT-0001",
  "vkg": "DEST",
  "question": "Which students are in Class 1?",
  "sample_type": "ONTOLOGY_ONLY",
  "sparql": "PREFIX ex:   <http://example.org/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\nSELECT ?student\nWHERE {\n  ?student a ex:Student ;\n           ex:inClass ex:Class1 .\n}\nORDER BY ?student"
}
```

### Best Practices
- Keep reference items close to source text; triples serve only as anchors for reproduction and filtering.
- If predicate/object is uncertain, use wildcard `*` as placeholder and refine later.
- In `HYBRID_MIXED` scenarios, use reference items to record supplementary information and VALUES fragments (type="sparql").

## Dataset Web UI (Simple Annotation/Browser)

A simple web interface for browsing, creating, and modifying samples:
- Path: `src/dataset/ui/` (backend `server.py` + frontend static pages `static/`)
- Data root directory: `resources/datasets/` (each dataset corresponds to a `.jsonl` file; one sample JSON per line)

### Getting Started

```bash
uvicorn src.dataset.ui.server:app --host 0.0.0.0 --port 8008 --reload
```

Open your browser and navigate to `http://localhost:8008/`.

### Features
- Dropdown to select `.jsonl` files under `resources/datasets/`; create new files
- Global settings for annotator name (required) and VKG Endpoint (for online SPARQL execution)
- Previous/Next buttons to browse samples; search and jump by `id`
- Form-based sample editing (`id/vkg/sample_type/question/sparql`); references edited via input controls (no JSON required)
- Multiple references supported: type/purpose/content + triples (NT format, one per line), auto-converted to JSON on save
- Save updates / append new samples / delete
- Run current sample's `sparql` and preview results (SELECT queries auto-parsed as table overview)

### Constraints
- Annotator name cannot be empty; automatically written to `annotator` field when saving samples
- SPARQL must be directly executable (include PREFIX)
