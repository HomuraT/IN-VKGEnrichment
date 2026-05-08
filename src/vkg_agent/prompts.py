# Simplified VKG Agent Prompts
# Designed for minimal LLM calls with structured JSON output

# =============================================================================
# Core Prompts (Used in simplified pipeline)
# =============================================================================

# Generate k candidate SPARQL queries in one call (FAST PATH - No analysis required)
generate_candidates_simple_json = """
You are an expert SPARQL generator. Generate {{ k }} executable SPARQL queries to answer the user's question.

## Context

Prefixes:
{{ prefixes_text }}

Ontology:
{{ ontology_text }}

Mappings:
{{ mappings_text }}

Data Samples:
{{ triples_text }}

## Question

"{{ question }}"

## Instructions

1. **Exploration First**: When uncertain, use exploration patterns as your first candidate:
   - **Schema Exploration**: `SELECT ?p ?o WHERE { ?s a <Class> . ?s ?p ?o } LIMIT 10` - discover class properties
   - **Entity Exploration**: `SELECT ?p ?o WHERE { <Entity> ?p ?o } LIMIT 10` - discover entity properties

2. **Candidate Strategy**: Candidate 1 = exploration query; Candidate 2-{{ k }} = direct answer attempts

3. **Rules**: Use only URIs from Context. Do NOT use EXISTS/NOT EXISTS.

## JSON Output

```json
{
  "candidates": [
    {"text": "Explore Company properties", "purpose": "Discover available properties", "query_type": "SELECT", "prefixes": ["npdv"], "body": "SELECT ?p ?o WHERE { ?s a npdv:Company . ?s ?p ?o } LIMIT 10"},
    {"text": "Get company names", "purpose": "Direct answer attempt", "query_type": "SELECT", "prefixes": ["npdv"], "body": "SELECT ?name WHERE { ?s a npdv:Company . ?s rdfs:label ?name }"}
  ]
}
```

Output ONLY valid JSON with ALL {{ k }} candidates. No analysis text.
"""

# Generate k candidate SPARQL queries in one call (SLOW PATH - With analysis)
generate_candidates_json = """
You are an expert SPARQL generator for a Virtual Knowledge Graph (VKG).

## Context

Prefixes:
{{ prefixes_text }}

Ontology:
{{ ontology_text }}

Mappings:
{{ mappings_text }}

Data Samples:
{{ triples_text }}

## Task

Generate {{ k }} diverse SPARQL candidates to answer: "{{ question }}"

## Rules

1. **Exploration First**: When uncertain, prioritize exploration patterns:
   - **Schema**: `SELECT ?p ?o WHERE { ?s a <Class> . ?s ?p ?o } LIMIT 10`
   - **Entity**: `SELECT ?p ?o WHERE { <Entity> ?p ?o } LIMIT 10`

2. **Query Types**: Use ASK/DESCRIBE for verification, SELECT for answers.

3. **Constraints**: Use only URIs from Context. Do NOT use EXISTS/NOT EXISTS.

## Response Format

**Step 1: Analysis** (Required):
- Identify key entities/concepts from the question
- Review ontology/mappings for relevant classes and properties
- Explain your strategy for each candidate

**Step 2: JSON Output** (at the END):

```json
{
  "candidates": [
    {"text": "Explore class properties", "purpose": "Discover available properties", "query_type": "SELECT", "prefixes": ["genex"], "body": "SELECT ?p ?o WHERE { ?s a genex:Gene . ?s ?p ?o } LIMIT 10"},
    {"text": "Direct answer", "purpose": "Attempt to answer", "query_type": "SELECT", "prefixes": ["genex"], "body": "SELECT ?gene WHERE { ?gene a genex:Gene }"}
  ]
}
```

**CRITICAL**: JSON must be at the END, contain ALL {{ k }} candidates, be valid and complete.
"""

# Generate k candidate SPARQL queries with iteration memory
generate_candidates_with_memory_json = """
You are an expert SPARQL generator. Previous queries failed. Generate {{ k }} NEW candidates.

## Context

Prefixes:
{{ prefixes_text }}

Ontology:
{{ ontology_text }}

Mappings:
{{ mappings_text }}

Data Samples:
{{ triples_text }}

Previous History:
{{ memory_text }}

## Task

Generate {{ k }} NEW candidates to answer: "{{ question }}"

## Correction Strategy

1. **Learn from Failures**: Do NOT reuse URIs/predicates that returned NO_ROWS or errors.

2. **Pivot with Exploration**: Use exploration patterns to discover what's actually available:
   - **Schema**: `SELECT ?p ?o WHERE { ?s a <Class> . ?s ?p ?o } LIMIT 10`
   - **Entity**: `SELECT ?p ?o WHERE { <Entity> ?p ?o } LIMIT 10`

3. **Constraints**: Use only URIs from Context. Do NOT use EXISTS/NOT EXISTS.

## Response Format

**Step 1: Analysis** (Required):
- Review failures: what URIs/predicates failed?
- Propose new strategies based on available context

**Step 2: JSON Output** (at the END):

```json
{
  "candidates": [
    {"text": "Explore class after failures", "purpose": "Discover actual properties", "query_type": "SELECT", "prefixes": ["genex"], "body": "SELECT ?p ?o WHERE { ?s a genex:Gene . ?s ?p ?o } LIMIT 10"},
    {"text": "Alternative approach", "purpose": "Try different predicate", "query_type": "SELECT", "prefixes": ["genex"], "body": "SELECT ?gene WHERE { ?gene a genex:Gene }"}
  ]
}
```

**CRITICAL**: JSON must be at the END, contain ALL {{ k }} candidates, be valid and complete.
"""

# Decide whether candidates answer the question, optionally refine
decide_and_refine_json = """
You are a SPARQL Evaluator. Analyze execution results and decide the next step.

## Question
{{ question }}

## Context

Ontology:
{{ ontology_text }}

Mappings:
{{ mappings_text }}

Triples:
{{ triples_text }}

## Execution Results
{{ exec_results_text }}

{% if memory_text %}
## Previous History
{{ memory_text }}
{% endif %}

{{ require_select_note }}

## Decision Logic

1. **YES**: Data returned that semantically answers the question. Columns must match user intent (not generic `?s ?p ?o`).

2. **REFINE**: Close but needs fixing (wrong URI, remove LIMIT, adjust columns). NEVER regress to generic exploration. Do NOT use EXISTS/NOT EXISTS (may cause VKG errors).

3. **FAIL**: No relevant data found or semantic mismatch. Provide insights on why.

## Response Format

**Step 1: Analysis** (Required):
- Check if results semantically answer the question
- Ignore success of generic exploration queries (`?p ?o`)
- Explain your decision reasoning

**Step 2: JSON Output** (at the END):

```json
{
  "decision": "YES" | "REFINE" | "FAIL",
  "selected_index": 1,
  "refined_query": {"prefixes": ["ns"], "body": "SELECT ..."},
  "insights": "Brief explanation"
}
```

**CRITICAL**:
1. JSON at the END, valid and complete
2. `selected_index` is 1-based (Candidate 1 → index 1)
3. YES: no `refined_query`; REFINE: must provide `refined_query`; FAIL: no `refined_query`
4. Return EXACTLY the columns user asked for (no more, no less)
5. Exploratory queries (`?s ?p ?o`) are NOT valid final answers
6. Do NOT use EXISTS/NOT EXISTS in refined queries (may cause VKG errors)
"""

# Best guess synthesis when all iterations fail
best_guess_json = """
You are a SPARQL Synthesizer. All iterations failed. Propose ONE final "Best Guess" query.

## Context

Prefixes:
{{ prefixes_text }}

Ontology:
{{ ontology_text }}

Mappings:
{{ mappings_text }}

Data Samples:
{{ triples_text }}

Previous History:
{{ history_text }}

## Task

Write a robust SPARQL query to answer: "{{ question }}"

## Requirements

1. **Avoid Failed Patterns**: Do NOT reuse URIs/predicates that returned NO_ROWS.
2. **Use Valid Prefixes**: Only use prefixes from the Context.
3. **Match User Intent**: Return exactly the columns user asked for (no `?s ?p ?o`).
4. **No EXISTS/NOT EXISTS**: These may cause VKG errors.

## Response Format

**Step 1: Analysis** (Required):
- Review failures and identify what went wrong
- Explain your synthesis strategy

**Step 2: JSON Output** (at the END):

```json
{
  "prefixes": ["prefix1", "prefix2"],
  "body": "SELECT ... WHERE { ... }",
  "rationale": "Why this query is likely to work"
}
```

**CRITICAL**: JSON at the END, valid and complete. Use only valid prefixes. No exploratory queries.
"""

# =============================================================================
# Textualization Prompts (Used for building knowledge base - keep these)
# =============================================================================

ontology_prompts = {
    "textualize_ontology_element_detailed": """
Please provide a comprehensive and detailed description of the following {{ element_type }} concept in natural language. Include its meaning, purpose, constraints, relationships, labels, comments, and all other important information. The description should be thorough and informative for knowledge retrieval.

{{ element_type }} data:
{{ element_data }}

Please provide a detailed and comprehensive description:
    """,
    
    "textualize_ontology_element_brief": """
Please provide a concise and brief description of the following {{ element_type }} concept in natural language. Focus on the core meaning and purpose in just 1-2 sentences. Keep it simple and clear for quick understanding.

{{ element_type }} data:
{{ element_data }}

Please provide a brief and concise description (1-2 sentences):
    """
}

vkg_mapping_prompts = {
    "textualize_vkg_mapping_detailed": """
Please provide a comprehensive and detailed description of the following VKG (Virtual Knowledge Graph) mapping in natural language. The mapping defines how data from a relational database is transformed into RDF triples. 

Include the following aspects in your description:
1. What is the purpose and domain of this mapping (what kind of data it handles)
2. What the SQL source query does (what data it extracts from which tables)
3. What the RDF target template produces (what kind of RDF triples are generated)
4. How the mapping transforms relational data into semantic knowledge graph format
5. What ontology concepts and properties are used
6. Any important details about the data transformation process

VKG Mapping data:
{{ mapping_data }}

Please provide a detailed and comprehensive description:
    """,
    
    "textualize_vkg_mapping_brief": """
Please provide a concise and brief description of the following VKG (Virtual Knowledge Graph) mapping in natural language. Focus on the core purpose and what data transformation it performs in just 2-3 sentences.

VKG Mapping data:
{{ mapping_data }}

Please provide a brief and concise description (2-3 sentences):
    """
}

aggregated_triples_prompts = {
    "textualize_aggregated_subject_detailed": """
Please provide a comprehensive and detailed natural language description of the following RDF subject and all its associated triples.

Include the following aspects:
1. What does this subject represent in the knowledge graph (entity type, role, semantics)
2. Summarize key relationships grouped by predicates (what properties it has and to what targets)
3. Highlight important attributes and relationships that define or characterize the subject
4. If possible, infer the overall meaning and context from the triples (without inventing facts)
5. Keep it factual, structured, and helpful for downstream retrieval

Subject data (JSON):
{{ subject_data }}

Please provide a detailed and comprehensive description:
    """,

    "textualize_aggregated_subject_brief": """
Please provide a concise and brief description (1-3 sentences) of the following RDF subject and its triples. Focus on the core meaning and most important relationships.

Subject data (JSON):
{{ subject_data }}

Please provide a brief and concise description (1-3 sentences):
    """
}

# =============================================================================
# Reasoning Explanation Prompt (Used to generate final reasoning explanation)
# =============================================================================

generate_reasoning_explanation_markdown = """
You are a VKG (Virtual Knowledge Graph) reasoning analyst. Based on the complete exploration process, explain why the final SPARQL query was generated (or why it failed).

## User Question
{{ question }}

## Final Decision
{{ final_decision }}
{% if final_sparql %}
Final SPARQL:
```sparql
{{ final_sparql }}
```
{% endif %}

## Exploration History
{{ memory_summary }}

Experience Trace:
{{ experience_trace }}

## Retrieved Context

### Ontology Concepts
{{ ontology_text }}

### VKG Mappings
{{ mappings_text }}

### Sample Triples
{{ triples_text }}

---

Please generate a comprehensive reasoning explanation in **Markdown format** with the following structure:

## Exploration Process

Provide a **detailed chronological account** of the exploration journey:
- **Round 1**: What initial strategies were attempted? What query types were used (ASK/DESCRIBE/CONSTRUCT/SELECT)? What were the results?
- **Round 2-N** (if applicable): How did the system adapt based on previous failures? What new strategies were explored? What insights from the Memory Bank guided these changes?
- **Challenges**: What specific obstacles were encountered (e.g., empty results, wrong URIs, missing predicates)? How were they addressed?
- **Adaptations**: How did query patterns evolve across iterations (e.g., switching from direct URI to label matching, adding REGEX filters, simplifying complex patterns)?

Be specific with examples: "In Round 1, the system attempted to use `obo:UBERON_0002372` directly, but this returned NO_ROWS..."

## Key Evidence from Context

Provide a **detailed analysis** of how the retrieved context contributed to the solution:
- **Ontology Concepts**: Which classes and properties were most relevant? How did their definitions and relationships inform the query design?
- **VKG Mappings**: Which mappings exposed the necessary data? What table-to-ontology transformations were crucial?
- **Sample Triples**: Which example triples demonstrated valid patterns? How did they validate or refute query strategies?

Reference specific examples: "The mapping `M1: SELECT anatEntityId FROM anatEntity` revealed that anatomical entities are stored in the `anatEntity` table..."

## Query Design Rationale

Provide a **detailed justification** for the final SPARQL query design:
- **URI Selection**: Why use direct URIs vs rdfs:label matching? What evidence supported this choice?
- **Pattern Construction**: Why use specific triple patterns? What relationships were being explored?
- **Filters and Constraints**: Why add REGEX, FILTER, or other constraints? What problems did they solve?
- **Query Type Choice**: Why SELECT vs ASK/CONSTRUCT/DESCRIBE? What made this the most appropriate?
- **Edge Cases**: How does the query handle potential data inconsistencies or missing values?

If the query failed: Explain what is missing (unmapped columns, incomplete ontology, data gaps) and what refinements would be needed.

## Summary

Provide a **brief conclusion** (2-3 sentences) summarizing:
- Whether the system successfully answered the question or why it failed
- The key insight or breakthrough that led to the solution (or the main obstacle if failed)
- One sentence on overall query quality or confidence

## Improvement Suggestions

Provide **concrete recommendations** to enhance the VKG (ontology or mappings) based on the exploration experience:

### 1. Missing Mappings

If the exploration revealed data in sample triples that lacks explicit mapping documentation:
- **Identify**: Which properties or relationships appeared in triples but were not documented in VKG Mappings?
- **Propose**: Write complete mapping definitions with `mappingId`, `target`, and `source`
- **Example Format**:

```
mappingId	Mapping:Discovery:Name
target		npd:discovery/{dscNpdidDiscovery} npdv:name {cmpLongName}^^xsd:string .
source		SELECT "dscNpdidDiscovery", "cmpLongName" FROM "discovery"
```

### 2. Datatype Issues

If queries failed or behaved unexpectedly due to datatype inconsistencies:
- **Identify**: Which properties have ambiguous or incorrect datatypes (e.g., years stored as strings)?
- **Propose**: Explicit datatype declarations in ontology + type casting in mappings
- **Example Format**:

```turtle
# Ontology Update: Standardize Year Datatype
npdv:discoveryYear a owl:DatatypeProperty ;
    rdfs:domain npdv:Discovery ;
    rdfs:range xsd:integer .
```

```
# Mapping Update: Add Type Casting
mappingId	Mapping:Discovery:Year
target		npd:discovery/{dscNpdidDiscovery} npdv:discoveryYear {dscDiscoveryYear}^^xsd:integer .
source		SELECT "dscNpdidDiscovery", CAST("dscDiscoveryYear" AS INTEGER) AS "dscDiscoveryYear" FROM "discovery" WHERE "dscDiscoveryYear" IS NOT NULL
```

### 3. Missing Ontology Concepts

If the question required classes or properties not found in the ontology:
- **Identify**: What concepts or relationships were needed but missing?
- **Propose**: New class/property definitions with labels, domains, and ranges
- **Example Format**:

```turtle
# Suggested New Datatype Property
npdv:operatorName a owl:DatatypeProperty ;
    rdfs:label "operator name"@en ;
    rdfs:domain npdv:ProductionLicence ;
    rdfs:range xsd:string ;
    rdfs:comment "The name of the company operating the production licence"@en .

# Suggested New Object Property
npdv:operatedBy a owl:ObjectProperty ;
    rdfs:label "operated by"@en ;
    rdfs:domain npdv:ProductionLicence ;
    rdfs:range npdv:Company ;
    rdfs:comment "Links a production licence to the company that operates it"@en .

# Suggested New Class
npdv:OffshoreField a owl:Class ;
    rdfs:label "offshore field"@en ;
    rdfs:subClassOf npdv:Field ;
    rdfs:comment "A petroleum field located in offshore waters"@en .
```

### 4. Incomplete Mappings

If mappings exist but lack necessary filters or joins:
- **Identify**: Which mappings returned irrelevant or incomplete data?
- **Propose**: Enhanced SQL queries with proper WHERE clauses, JOINs, or aggregations
- **Example Format**:

```
mappingId	Mapping:Discovery:Active
target		npd:discovery/{dscNpdidDiscovery} a npdv:Discovery .
source		SELECT "dscNpdidDiscovery" FROM "discovery" WHERE "dscStatus" = 'ACTIVE' AND "dscNpdidDiscovery" IS NOT NULL
```

### 5. Label and Documentation Gaps

If URIs were hard to interpret or queries failed due to missing labels:
- **Identify**: Which classes/properties lack `rdfs:label` or `rdfs:comment`?
- **Propose**: Add multilingual labels and descriptive comments
- **Example Format**:

```turtle
# Add Missing Labels
npdv:Company rdfs:label "Company"@en ;
    rdfs:comment "An organization involved in petroleum activities"@en .
```

**Guidelines for Suggestions**:
- Only propose changes if there is **clear evidence** from the exploration process (failed queries, missing data, type errors)
- Provide **complete, executable code snippets** in Turtle syntax
- Prioritize changes that would directly solve the encountered problems
- If no improvements are needed, state: "No specific improvements identified. The current VKG structure adequately supports this query."

## Output Format

Your response should be a comprehensive Markdown document with rich formatting:
- Use headings (##, ###), lists, `inline code`, **bold**, and *italic* for structure and emphasis
- Reference specific URIs, predicates, query patterns, and code snippets with technical precision
- Provide concrete examples from the ontology, mappings, and triples to support your analysis
- Write a thorough, detailed explanation that helps readers understand the complete reasoning process
- Include **complete, copy-pastable code snippets** in the Improvement Suggestions section

Output the Markdown text directly as plain text. Start immediately with the first ## heading.
"""