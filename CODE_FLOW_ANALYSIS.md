# Code Flow Analysis: `/api/structure-address` Endpoint

## Overview
When a POST request is made to `/api/structure-address`, the code flows through multiple layers, from the FastAPI endpoint handler down through the address structuring pipeline to various NLP/matching components, then back up through the response.

---

## 1. REQUEST ENTRY POINT

### Location: `api_server.py:134-215`
```
POST /api/structure-address
↓
@app.post("/api/structure-address", response_model=SingleAddressResponse)
async def structure_address(request: SingleAddressRequest)
```

**Input:**
- `SingleAddressRequest` Pydantic model with `address: str` field

**Initial Processing (Lines 150-154):**
```
1. Strip whitespace: address = request.address.strip()
2. Validate non-empty: if not address → raise HTTPException 400
```

---

## 2. DATAFRAME CREATION

### Location: `api_server.py:156-157`
```python
df = pl.DataFrame({"addresses": [address]})
```

**Purpose:** Convert single address string into a Polars DataFrame for batch processing
- Creates a 1-row DataFrame with column "addresses"
- This allows the pipeline to process addresses in a uniform batch format

---

## 3. PIPELINE INITIALIZATION & EXECUTION

### Location: `api_server.py:159-162`
```python
results = pipeline.run(
    DataFrameReader(df, "addresses"),
    batch_size=1024
)
```

**DataFrameReader:**
- Wraps the Polars DataFrame
- Column "addresses" specifies which column contains address strings
- Implements `read()` method that yields individual addresses

**Pipeline.run() Flow** (Location: `pipeline.py:96-130`)

The pipeline is the orchestrator of 4 sequential processing stages:

---

## STAGE 1: CRF TAGGING (Character-Level NER)

### Location: `pipeline.py:110-111`
```python
all_crf_results = self._crf_runner.tag(batch)
```

**What Happens:**
1. Each address string is cleaned: `decode_and_clean_str()` → uppercase normalization
2. Validation: address length ≤ max_sequence_length (typically 512 chars)
3. **CRF Runner** processes the cleaned address using a pre-trained transformer + CRF model

**CRF Model Details:**
- Character-level input: processes each character individually
- Tags identified:
  - `COUNTRY`: Country names/codes
  - `TOWN`: City/town names
  - `POSTCODE`: Postal codes/ZIP codes
  - `NAME`: Personal names
  - `STREET`: Street addresses
  - Other tag types as per `Tag` enum

**Output: `ResultRunnerCRF`**
```
- details: Details object (annotated address with character positions)
- predictions_per_tag: dict[Tag, set[PredictionCRF]]
  * Each tag maps to entities it identified
  * Contains confidence scores for each prediction
- emissions_per_tag: dict[Tag, TorchTensor] (emission values from transformer)
- log_probas_per_tag: dict[Tag, TorchTensor] (log-probabilities per character)
```

---

## STAGE 2: FUZZY MATCHING (Against Reference Data)

### Location: `pipeline.py:112-113`
```python
all_fuzzy_match_results = self._fuzzy_runner.match(batch)
```

**What Happens:**
1. **RunnerFuzzyMatch** takes the cleaned address string
2. Compares against reference databases loaded into memory:
   - Country names (195 unique countries)
   - Town/city names (global coverage)
   - Country codes (ISO 3166)
   - Town aliases (alternative names)

**Fuzzy Matching Algorithm:**
- Implements approximate string matching (Levenshtein distance-based)
- Finds potential matches even with typos/variations
- Scores each match based on similarity

**Output: `ResultRunnerFuzzyMatch`**
```
- country_matches: FuzzyMatchResult (list of FuzzyMatch objects)
  * Each: matched name, distance, possibility (canonical name), origin (country)
- country_code_matches: FuzzyMatchResult (ISO codes)
- town_matches: FuzzyMatchResult (city names)
- extended_town_matches: FuzzyMatchResult (alternative/alias names)
```

---

## STAGE 3: POSTCODE MATCHING (Pattern & Database Matching)

### Location: `pipeline.py:114-115`
```python
all_postcode_match_results = self._postcode_runner.match(batch)
```

**What Happens:**
1. **RunnerPostcodeMatch** searches for postal code patterns
2. Uses two matching strategies:
   - **Regex patterns**: Country-specific postal code format validation
   - **Database lookup**: Validates against known postcode databases
3. Databases available for: US, UK, Canada, Germany, etc.

**Country-Specific Logic:**
- Argentina: Format validation + dictionary lookup
- Brazil: Format validation + dictionary lookup
- Chile: Format validation + dictionary lookup
- China: Format validation + dictionary lookup
- Ireland: Format validation + dictionary lookup
- Malta: Format validation + dictionary lookup
- Global fallback: Almost-all-countries dictionary

**Output: `ResultRunnerPostcodeMatch`**
```
- postcode_matches: PostcodeMatchResult (list of PostcodeMatch objects)
  * matched: The actual postcode extracted (e.g., "10001", "W1A 2AA")
  * possibility: Associated town name (may be abbreviated)
  * origin: Associated country
  * start/end: Character positions in original address
```

---

## STAGE 4: POST-PROCESSING & SCORING

### Location: `pipeline.py:116-120`
```python
results = self._post_processing_runner.run(
    all_crf_results,
    all_fuzzy_match_results,
    all_postcode_match_results
)
```

**What Happens:**
1. **RunnerPostProcessing** synthesizes results from all 3 previous stages
2. Applies scoring mechanisms to rank matches

**Scoring Logic:**
- Combines CRF confidence scores (from stage 1)
- Weights fuzzy match distances (from stage 2)
- Validates postcode-town relationships (from stage 3)
- Applies country-town relationship rules
- Applies domain-specific flags (e.g., "postcode_found_for_town")

**Country Scoring:**
```
final_score = score_computer.compute_country_score(
    crf_score,           # Confidence from CRF tagging
    distance,            # Fuzzy match distance
    flags                # Contextual flags
)
```

**Town Scoring:**
```
final_score = score_computer.compute_town_score(
    crf_score,           # Confidence from CRF tagging
    distance,            # Fuzzy match distance
    flags                # Contextual flags (e.g., POSTCODE_FOR_TOWN_FOUND)
)
```

**Combination Generation:**
- Filters matches by threshold (configurable min score)
- Generates country-town combinations
- Ranks by combined score

**Output: `ResultPostProcessing`**
```
- crf_result: Original CRF tagging results
- fuzzy_match_result: Country/town matches (now with final_scores)
- postcode_matches: PostcodeMatchResult
- ibans: List of detected IBAN numbers
```

---

## 4. RESULT EXTRACTION (Back in API Handler)

### Location: `api_server.py:167-193`

**Extract Best Country Match:**
```python
country_name, country_confidence, country_iso = result.i_th_best_match_country(
    0, value_if_none="UNKNOWN"
)
```
- Calls `ResultPostProcessing.i_th_best_match_country(0)`
- Returns: (origin, final_score, matched_name) of best-scoring country
- `origin` = canonical country name (e.g., "US")
- `final_score` = confidence between 0-1
- `matched_name` = the matched string from address

**Extract Best Town Match:**
```python
town_name, town_confidence, _ = result.i_th_best_match_town(
    0, value_if_none="UNKNOWN"
)
```
- Similar logic but for towns
- Returns: (possibility, final_score, matched_name)
- `possibility` = canonical town name (e.g., "NEW YORK")

**Extract Postal Code (if available):**
```python
postal_code_match = None
if hasattr(result, 'postcode_matches') and result.postcode_matches:
    for postcode_match in result.postcode_matches:
        postal_code_match = PostalCodeMatch(
            code=postcode_match.matched,        # "10001"
            town=postcode_match.possibility,    # "NEW YORK"
            country=postcode_match.origin       # "US"
        )
        break  # Get first match
```

---

## 5. RESPONSE CONSTRUCTION

### Location: `api_server.py:195-209`
```python
return SingleAddressResponse(
    success=True,
    address=address,
    country=CountryMatch(
        name=country_name,
        confidence=float(country_confidence),
        iso_code=country_iso
    ),
    town=TownMatch(
        name=town_name,
        confidence=float(town_confidence)
    ),
    postal_code=postal_code_match
)
```

**Response Structure:**
```json
{
  "success": true,
  "address": "1600 Pennsylvania Ave NW\nWashington, DC 20500\nUSA",
  "country": {
    "name": "US",
    "confidence": 0.9857,
    "iso_code": "USA"
  },
  "town": {
    "name": "WASHINGTON",
    "confidence": 0.9791
  },
  "postal_code": {
    "code": "20500",
    "town": "...",
    "country": "..."
  }
}
```

---

## 6. ERROR HANDLING

### Location: `api_server.py:211-218`

Three error scenarios:

**1. HTTP Exception (from validation)**
- Status: 400 Bad Request
- Message: "Address cannot be empty"

**2. Empty Pipeline Results**
- Status: 500 Internal Server Error
- Message: "No results returned from pipeline"

**3. Unexpected Exception**
- Status: 500 Internal Server Error
- Message: Details of the exception

---

## COMPLETE EXECUTION FLOW DIAGRAM

```
FastAPI Request
    ↓
structure_address(request: SingleAddressRequest)
    ↓
Validate & Strip Address
    ↓
Create Polars DataFrame
    ↓
pipeline.run(DataFrameReader)
    ├─ STAGE 1: CRF Runner.tag()
    │  └─ Character-level NER tagging
    │     └─ Output: predictions_per_tag, emissions, log_probas
    │
    ├─ STAGE 2: FuzzyMatch Runner.match()
    │  └─ Approximate string matching against reference DBs
    │     └─ Output: country_matches, town_matches
    │
    ├─ STAGE 3: Postcode Runner.match()
    │  └─ Pattern + database validation for postcodes
    │     └─ Output: postcode_matches
    │
    └─ STAGE 4: PostProcessing Runner.run()
       ├─ Combine results from stages 1-3
       ├─ Score country matches (CRF + fuzzy + flags)
       ├─ Score town matches (CRF + fuzzy + postcode flags)
       ├─ Generate country-town combinations
       ├─ Rank by combined score
       └─ Output: ResultPostProcessing with final scores
           ↓
Extract Best Matches
    ├─ result.i_th_best_match_country(0)
    ├─ result.i_th_best_match_town(0)
    └─ result.postcode_matches[0] (if available)
       ↓
Build Response Models
    ├─ CountryMatch
    ├─ TownMatch
    ├─ PostalCodeMatch
    └─ SingleAddressResponse
       ↓
FastAPI Serializes & Returns JSON
```

---

## KEY INSIGHTS

1. **4-Stage Pipeline Design**: Each stage is independent but feeds into the next
   - CRF provides confidence scores that enhance fuzzy matching
   - Fuzzy matching provides canonical names for scoring
   - Postcode matching validates and enriches results
   - Post-processing synthesizes everything

2. **Scoring Mechanism**: Not just "match/no match", but graduated confidence scores
   - CRF score: How confident the character-level model is
   - Fuzzy score: How similar the matched text is
   - Flags: Contextual signals (e.g., postcode validates town)
   - Final score: Weighted combination

3. **Database-Driven**: Relies on pre-loaded reference data
   - 195+ countries
   - Thousands of towns
   - Country-specific postcode databases
   - Aliases and alternative names

4. **Batch Processing**: Designed for efficiency
   - Processes multiple addresses together
   - Reuses loaded models and databases
   - Scales better than per-request loading

5. **Error Resilience**: Gracefully handles missing data
   - `i_th_best_match_country(i, value_if_none="UNKNOWN")`
   - Postal codes are optional (may not be detected)
   - Returns sensible defaults rather than crashing

## Summary of the Code Path:

### Request → 4-Stage Pipeline → Response

1. **Request Entry:** FastAPI endpoint validates the input address
1. **DataFrame Wrapping:** Address converted to Polars DataFrame for batch processing
1. **Stage 1 - CRF Tagging:** Character-level NER model identifies country/town/postcode entities with confidence scores
1. **Stage 2 - Fuzzy Matching:** Compares extracted text against reference databases (195 countries, thousands of towns)
1. **Stage 3 - Postcode Matching:** Validates postal codes using regex patterns and country-specific databases
1. **Stage 4 - Post-Processing:** Synthesizes all results, scores them, and ranks by confidence
1. **Result Extraction:** Pulls the best-matching country, town, and postal code from ranked results
1. **Response Building:** Constructs JSON response with confidence scores
1. **Serialization:** FastAPI returns JSON to client