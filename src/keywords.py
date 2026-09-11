"""Search-term lexicon for the Gasoline Price Search (GPS) index.

Shared by 01_data_collection and 02_index_construction so both notebooks
always use the same 61 terms. 

Construction (keyword list finalized 27 Aug 2026):
- Start from 8 ground keywords taken from the top 25 related queries of
  "Gasoline" (Google category: Fuel): gas, gas station, gas prices,
  gas mileage, gas tank, gasoline, gas stations, gas price.
- Drop related queries containing "near me", a byproduct of Google's
  location service launched in 2015.
- Drop duplicates and irrelevant queries: e.g. "shell" (too general and broad),
  "natural gas" and "air gas" (not about gasoline), "toyota" (too broad),
  and "bp".
"""

# Base term for pairwise rescaling; every other term is retrieved together with it.
ANCHOR = "gas price"

KEYWORDS = [
    "average gas prices",
    "average price of gas",
    "best gas mileage",
    "best gas mileage car",
    "best gas mileage cars",
    "best gas prices",
    "best suv gas mileage",
    "ca gas prices",
    "california gas price",
    "california gas prices",
    "car gas mileage",
    "cheap gas",
    "cheap gas prices",
    "costco gas",
    "costco gasoline",
    "current gas price",
    "gallon of gas price",
    "gas buddy",
    "gas calculator",
    "gas costco price",
    "gas mileage",
    "gas mileage calculator",
    "gas mileage for cars",
    "gas price",
    "gas price average",
    "gas price news",
    "gas price per gallon",
    "gas price today",
    "gas prices",
    "gas prices costco",
    "gas prices in california",
    "gas prices news",
    "gas prices today",
    "gas station prices",
    "gasoline can",
    "gasoline engine",
    "gasoline price",
    "gasoline prices",
    "gasoline tax",
    "high gas prices",
    "jeep gas mileage",
    "mileage calculator",
    "mobil",
    "ohio gas prices",
    "price of gas",
    "price of gasoline",
    "sam's club gas",
    "sam's club gas price",
    "sam's club gas prices",
    "sam's gas price",
    "sam's gas prices",
    "sams gas",
    "sams gas price",
    "shell gas",
    "speedway",
    "subaru gas mileage",
    "us gas prices",
    "walmart gas",
    "walmart gas price",
    "what is gasoline",
    "what is in gasoline",
]

# Sanity checks
assert len(KEYWORDS) == 61, f"expected 61 keywords, got {len(KEYWORDS)}"
assert len(set(KEYWORDS)) == len(KEYWORDS), "duplicate keyword in KEYWORDS"
assert ANCHOR in KEYWORDS, "ANCHOR must be one of the KEYWORDS"
