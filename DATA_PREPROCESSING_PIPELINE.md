# Data Preprocessing Pipeline: Geonames & Postal Codes

## Overview

The system uses **preprocessing scripts** to transform raw geographic data into optimized formats for fast fuzzy matching. These scripts run **offline** during development/setup and generate compressed JSON files that are loaded at runtime.

---

## Raw Data Sources

### Location: `/resources/raw/`

```
resources/raw/
├── geonames/
│   ├── allCountries.txt          (1.7 GB Raw geonames towns/cities data)
│   ├── allCountries.zip          (Compressed)
│   ├── countryInfo.txt           (Country metadata)
│   ├── alternateNamesV2/
│   │   └── alternateNamesV2.txt (762.1 MB Language-specific alternate names)
|   |   |__ iso-languagecodes.txt
│   └── alternateNamesV2.zip
├── postcodes/
│   └── allCountries.txt (140 MB Country-specific postcode data)
└── restCountries/
    └── (REST Countries API data for postal code specs)
```

**Sources:**
- **Geonames**: http://download.geonames.org/
- **Postcodes**: OpenDataSoft Geonames Postcodes
- **REST Countries**: https://restcountries.com/

---

## Preprocessing Scripts

### 1. `preprocess_geonames_countries.py`

**Purpose:** Extract and normalize country data from Geonames

**Input:**
- `countryInfo.txt` - Country metadata (ISO codes, names, languages)
- `alternateNamesV2/` - Alternate names in multiple languages

**Processing:**
1. Extract country ISO codes and names
2. Normalize names to uppercase
3. Extract alternate names from multiple languages
4. Generate country name aliases

**Output:**
- `resources/country_names.json` - Compressed country data with aliases
- `resources/country_aliases.json` - Alternative names for fuzzy matching

**Key Functions:**
```python
- get_country_data()              # Load country metadata
- get_country_languages()          # Extract language preferences per country
- filter_and_save_alternate_names()  # Process alternate names
```

---

### 2. `preprocess_geonames_towns.py`

**Purpose:** Extract, filter, and normalize town/city data from Geonames

**Input:**
- `allCountries.txt` - All geonames entries (2.7M+ rows)
- Feature code filters from config

**Processing:**
1. Parse tab-separated geonames data
2. Filter by feature code (PPLA, PPLC = administrative centers, capitals)
3. Filter by population (minimum thresholds per feature type)
4. Normalize town names (uppercase, remove special chars)
5. Extract alternate names per language
6. Remove duplicates per country-town pair

**Output:**
- `resources/town_aliases.json` - Compressed town data with aliases per country

**Key Filtering:**
```python
Feature Codes (configured in DatabaseConfig):
- PPLA: Administrative division (state/province level)
- PPLC: Capital of a political entity
- PPL:  City/populated place (fallback)

Population Thresholds (configurable):
- Capitals: ≥5,000 people (or all if under threshold)
- Major cities: ≥15,000 people
- Default: Configurable minimum
```

**Key Functions:**
```python
- filter_and_save_alternate_names()  # Extract and normalize
- ALLCOUNTRIES_COLUMN_NAMES         # Schema definition
- get_country_overrides()           # Country-specific rules
```

---

### 3. `preprocess_geonames_postcodes.py`

**Purpose:** Create country-specific postcode databases and regex patterns

**Input:**
- Raw postcode files per country:
  - `postcodes/Argentina.txt`
  - `postcodes/Brazil.txt`
  - `postcodes/Chile.txt`
  - `postcodes/China.txt`
  - `postcodes/Ireland.txt`
  - `postcodes/Malta.txt`
  - Global: `postcodes/allcountries_postcodes.txt`

**Processing:**
1. Parse postcode data (country code, postcode, place name, lat/long)
2. Generate regex patterns for postcode format validation
3. Normalize postcode format (uppercase, remove special chars)
4. Map postcodes to towns and countries
5. Create lookup dictionaries and regex lists
6. Compress with zlib for storage efficiency

**Regex Generation Example:**
```
Postcode format: "10001"
↓
Regex pattern: "[0-9][0-9][0-9][0-9][0-9]"

Postcode format: "SW1A 2AA" (UK)
↓
Regex pattern: "[A-Z][A-Z][0-9][A-Z] [0-9][A-Z][A-Z]"
```

**Countries with Special Processing:**
```python
COUNTRIES_WITH_SPECIAL_POST_CODES = ['AR', 'BR', 'CL', 'CN', 'IE', 'MT']
```

**Output (per country):**
- `resources/post_codes/{COUNTRY}_dict.json` - Lookup dictionary
  ```json
  {
    "10001": ["WASHINGTON", "US"],
    "10002": ["NEW YORK", "US"],
    ...
  }
  ```

- `resources/post_codes/{COUNTRY}_regex_list.json` - Regex patterns
  ```json
  [
    {"postcode": "10001", "regex": "[0-9]{5}"},
    {"postcode": "10002", "regex": "[0-9]{5}"},
    ...
  ]
  ```

**Key Functions:**
```python
filter_and_clean_dataframe()   # Clean and normalize
preprocess_and_save()         # Create dicts and regexes
CHAR_TO_REGEX                 # Character to regex mapping
```

---

### 4. `preprocess_rest_countries.py`

**Purpose:** Extract postal code specifications from REST Countries API

**Input:**
- REST Countries JSON data (from API or downloaded file)

**Processing:**
1. Parse REST Countries data
2. Extract postal code regex per country
3. Extract domain extensions (TLDs)
4. Extract phone number prefixes
5. Compress into single file

**Output:**
- `resources/country_specs.json` - Country specifications with regex patterns

**Example:**
```json
{
  "US": {
    "domain_extensions": [".us"],
    "postal_code_regex": "^\\d{5}(-\\d{4})?$",
    "phone_prefixes": ["+1"]
  },
  "GB": {
    "domain_extensions": [".uk"],
    "postal_code_regex": "^[A-Z]{1,2}\\d{1,2}[A-Z]?\\s?\\d[A-Z]{2}$",
    "phone_prefixes": ["+44"]
  }
}
```

**Key Functions:**
```python
- orjson for efficient JSON parsing
- zlib.compress() for storage
```

---

## Data Compression & Storage

All preprocessed data is **compressed with zlib** for efficient storage:

```python
import zlib
import orjson

# During preprocessing (write)
compressed = zlib.compress(orjson.dumps(data))

# During runtime (load in Database class)
data = orjson.loads(zlib.decompress(compressed_bytes))
```

**Benefits:**
- 70-80% size reduction
- Fast decompression at runtime
- Compact distribution

---

## Runtime Data Loading

### Location: `data_structuring/components/database.py`

The `Database` class loads all preprocessed data at initialization:

```python
class Database:
    def __init__(self, config: DatabaseConfig):
        # Load compressed datasets
        self.country_names = self._load_compressed_json(
            config.country_names)
        
        self.town_names = self._load_compressed_json(
            config.town_aliases)
        
        self.postcode_dicts = {
            'US': self._load_compressed_json(
                config.post_codes_us_dict),
            'GB': self._load_compressed_json(
                config.post_codes_gb_dict),
            # ... more countries
        }
        
        self.postcode_regexes = {
            'US': self._load_regex_patterns(
                config.post_codes_us_regex_list),
            # ... more countries
        }
        
        self.country_specs = self._load_compressed_json(
            config.country_specs)
    
    def _load_compressed_json(self, path):
        with open(path, 'rb') as f:
            return orjson.loads(zlib.decompress(f.read()))
```

---

## Data Flow Diagram

```
Raw Data Sources
├── Geonames (http://download.geonames.org/)
│   ├── allCountries.txt (2.7M rows)
│   ├── countryInfo.txt
│   └── alternateNamesV2/ (15M+ names)
│
├── Postcodes (OpenDataSoft)
│   └── Country-specific files
│
└── REST Countries (API)
    └── Country specifications

        ↓ PREPROCESSING SCRIPTS ↓

Preprocessing Phase (Offline)
├── preprocess_geonames_countries.py
│   └── Outputs: country_names.json, country_aliases.json
│
├── preprocess_geonames_towns.py
│   └── Outputs: town_aliases.json
│
├── preprocess_geonames_postcodes.py
│   └── Outputs: {country}_dict.json, {country}_regex_list.json
│
└── preprocess_rest_countries.py
    └── Outputs: country_specs.json

        ↓ COMPRESSION ↓

resources/
├── country_names.json (compressed)
├── country_aliases.json (compressed)
├── town_aliases.json (compressed)
├── country_specs.json (compressed)
└── post_codes/
    ├── us_dict.json (compressed)
    ├── us_regex_list.json (compressed)
    ├── gb_dict.json (compressed)
    └── ... (more countries)

        ↓ RUNTIME ↓

Database class loads all at initialization
↓
Runners use cached data for:
  - FuzzyMatch runner (fuzzy string matching)
  - Postcode runner (postcode validation)
  - Post-processing runner (relationship validation)
```

---

## Configuration

All preprocessing is controlled by configuration classes:

**`DatabaseConfig`** (`data_structuring/config.py`)
```python
class DatabaseConfig:
    # Input paths
    country_names = Path("resources/country_names.json")
    town_aliases = Path("resources/town_aliases.json")
    
    # Postcode databases per country
    post_codes_us_dict = Path("resources/post_codes/us_dict.json")
    post_codes_gb_dict = Path("resources/post_codes/gb_dict.json")
    # ... more countries
    
    # Filtering parameters
    geonames_country_feature_code_filter = {
        'PCLI': 10000,  # Country (min 10k people)
        'PCLH': 10000,  # Historical country
    }
    
    geonames_town_feature_code_filter = {
        'PPLA': 5000,   # State capital (min 5k)
        'PPLC': 5000,   # National capital (min 5k)
        'PPL': 15000,   # City (min 15k)
    }
    
    town_minimal_population = 5000  # Fallback threshold
```

**`PreprocessCLIConfig`** - CLI arguments for running scripts
```python
class PreprocessCLIConfig:
    input_geonames_all_countries_path: Path
    input_geonames_alternate_names_path: Path
    input_geonames_country_info_path: Path
    geonames_additional_languages: list[str] = ["en"]
```

---

## Running Preprocessing (One-Time Setup)

```bash
# Countries
python data_structuring/preprocessing/preprocess_geonames_countries.py \
    --input-geonames-country-info-path resources/raw/geonames/countryInfo.txt \
    --input-geonames-alternate-names-path resources/raw/geonames/alternateNamesV2/ \
    --geonames-additional-languages en

# Towns
python data_structuring/preprocessing/preprocess_geonames_towns.py \
    --input-geonames-all-countries-path resources/raw/geonames/allCountries.txt \
    --input-geonames-alternate-names-path resources/raw/geonames/alternateNamesV2/ \
    --input-geonames-country-info-path resources/raw/geonames/countryInfo.txt \
    --geonames-additional-languages en

# Postcodes
python data_structuring/preprocessing/preprocess_geonames_postcodes.py \
    --input-postcodes-directory resources/raw/postcodes/ \
    --postcode-countries AR BR CL CN IE MT

# Country Specs
python data_structuring/preprocessing/preprocess_rest_countries.py \
    --input-rest-countries-path resources/raw/restCountries/countries.json
```

---

## Summary

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `preprocess_geonames_countries.py` | Geonames metadata | `country_names.json` | Country name extraction & aliases |
| `preprocess_geonames_towns.py` | Geonames all entries | `town_aliases.json` | City/town extraction & filtering |
| `preprocess_geonames_postcodes.py` | Country postcode files | `{country}_dict.json`, `{country}_regex_list.json` | Postcode validation & lookup |
| `preprocess_rest_countries.py` | REST Countries API | `country_specs.json` | Postal code regex patterns |

**Key Points:**
- ✅ All preprocessing is **offline** (runs once during setup)
- ✅ Data is **compressed** (70-80% size reduction)
- ✅ Loaded **once at startup** into Database class
- ✅ Used by **Fuzzy Match**, **Postcode**, and **Post-Processing** runners
- ✅ Enables **fast, accurate** address matching at runtime
