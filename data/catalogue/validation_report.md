# Validation report

- Importer CSV: 86 records across six manufacturers; exact nine-column header confirmed.
- Manufacturer counts: Michelin 13, Bridgestone 20, Continental 7, Goodyear 13, Apollo 18, MRF 15.
- Required fields, database string-length limits, case-insensitive manufacturer/model duplicate check and one-source-row-per-product check: passed.
- Source-verified nominal dimensions: 104 across 40 product models in verified_sizes.csv; syntax and duplicate checks passed.
- Importer CSV sizes: intentionally blank. The importer turns each nominal dimension into a TyreSize row with ORM defaults load_index=91 and speed_rating=V. Those ratings were not verified for most listed dimensions and would be misleading.
- Image paths: all blank and accepted. The manifest documents each omission; no image files or hotlinks are referenced.
- Isolated import of the deliverable CSV: 86 tyre rows, 0 size rows, 0 tread-pattern rows.
- Disposable size-parser fixture: 205/55R16 and 225/45 R17 parsed to the expected width/aspect/rim integers. The schema then assigned the unverified 91/V defaults, confirming why verified size data is kept outside the importer CSV.
- Re-import duplicate behavior: 0 new rows and 86 duplicate warnings. Manufacturer/model uniqueness is application-level only; there is no database uniqueness constraint.
- Database used: disposable in-memory SQLite only. The configured application database was not opened or modified.

## Why some manufacturers have fewer than 20

- Michelin (13): the India product-family pages supplied 13 distinct passenger/SUV names in the selected scope.
- Bridgestone (20): the 20-record target was reached using India product pages and size catalogues.
- Continental (7): seven products had direct India-localized product-page or India SUV-guide evidence; other-market-only products were excluded.
- Goodyear (13): the India tyre catalogue lists 13 specific named models in the selected passenger/SUV scope; generic family labels were not counted as models.
- Apollo (18): 18 distinct car/SUV/van products were listed by Apollo India or its official store; size SKUs were not counted as separate models.
- MRF (15): 15 distinct passenger/SUV pattern names were independently supported by MRF India product pages/finder entries; size variants were not counted as new models.
