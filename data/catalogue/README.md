# TreadVision India tyre catalogue

This deliverable contains product records supported by official manufacturer India pages/catalogues. It includes up to 20 distinct product patterns per manufacturer; no separate models were invented to fill a target count. Current online listings are not a guarantee of live retail stock.

## Files

- treadvision_india_catalogue.csv — importer-compatible product records.
- source_tracking.csv — one record per product, fields supported, source URL(s), access date and confidence notes.
- verified_sizes.csv — 104 nominal dimensions verified from official sources.
- image_manifest.csv — image status for every product.
- validation_report.md — data checks and isolated importer results.

## Importer and database schema

The importer’s required columns are manufacturer, model, pattern_name, and category. The deliverable uses its documented nine-column order: manufacturer, model, pattern_name, category, construction, description, image_path, pattern_type, sizes.

Tyre column constraints from the SQLAlchemy model:
- manufacturer: VARCHAR(100), NOT NULL
- model: VARCHAR(150), NOT NULL
- pattern_name: VARCHAR(150), NOT NULL
- category: VARCHAR(80), NOT NULL
- construction: VARCHAR(50), default Radial, nullable unless supplied
- description: TEXT, nullable

There are no enum or CHECK constraints for category, construction, or pattern_type. Category accepts free text up to 80 characters. Construction accepts free text up to 50 characters; the ORM default is Radial, but this CSV has an explicit blank construction cell, which the importer passes through as an empty string.

TreadPattern has a required image_path VARCHAR(255), a unique foreign-key tyre_id, and pattern_type VARCHAR(80) with an ORM default of Asymmetric. The model comment lists Symmetrical, Asymmetric, and Directional as intended pattern_type values, but neither the database nor importer enforces that list. The importer creates a pattern row only when image_path is nonblank; the UI checks whether the stored path exists relative to the process working directory.

TyreSize requires integer width, aspect_ratio, and rim_diameter values and a foreign key to tyres. There is no uniqueness constraint for a size row. load_index is VARCHAR(20) with default 91 and speed_rating is VARCHAR(10) with default V. Since the importer discards those details from its sizes string and the ORM inserts 91/V, verified dimensions are kept in verified_sizes.csv instead of the importer CSV.

## Data choices

pattern_name repeats the manufacturer's published product/pattern designation. Where the manufacturer does not publish a separate tread-pattern name, no new geometric name was invented. construction and pattern_type are blank unless explicitly verified; product pages often describe compounds or applications but do not state radial/bias construction or a named symmetry class for the model.

The importer requires only manufacturer, model, pattern_name, and category. Category values are free text; there is no accepted-value enum. The database fields have string-length limits, but construction and pattern_type are not checked against a list. The database does not enforce manufacturer/model uniqueness.

The importer CSV sizes column is intentionally blank. Although product dimensions are verified and preserved in verified_sizes.csv, the importer discards load index and speed rating from each size. The ORM then silently assigns load index 91 and speed rating V to every imported size. Those values are unverified for most products. Keeping the importer CSV blank prevents incorrect 91/V data from entering the catalogue. verified_sizes.csv records nominal width/aspect/rim dimensions and source URLs separately.

Descriptions are short source-backed summaries where product pages gave a model-specific statement; otherwise the cell is blank. No construction details, tread geometry or fitments were inferred.

## Importer behavior confirmed

The importer accepts the nine columns in the catalogue header; extra columns are ignored, but none are included. It trims required text and skips existing records when the trimmed manufacturer and model match exactly (case-sensitive), reporting a warning for each duplicate. The database has no unique constraint for this pair.

image_path may be blank; the importer then creates no tread-pattern row. It does not check nonblank image paths for file existence. Size lists, when supplied, are comma-separated, for example 205/55R16,225/45R17. The parser tolerates spaces and stores only integer width/aspect/rim. Invalid size tokens are silently skipped. The schema supplies the unverified 91/V defaults noted above. pattern_type is not validated; the model comment names Symmetrical, Asymmetric and Directional as intended values. A pattern row is only created when image_path is nonblank.

## Images

No image files are bundled because official product images are visible online but redistribution terms were not confirmed. The manifest documents the gap per product. All image_path values are blank; no fabricated paths or hotlinks are used.

## Why some brands have fewer than 20 records

Michelin has 13 distinct names listed across the official India passenger/SUV family pages. Continental has 7 products with direct India-localized product-page or India SUV-guide evidence; products documented only for other markets were excluded. Goodyear has 13 specific model names in its India tyre catalogue; generic family labels were excluded. MRF has 15 independently named passenger/SUV patterns supported by India product pages/finder entries. Apollo has 18 distinct car/SUV/van products listed by Apollo India or its official store. Bridgestone reached the 20-record target. In every case, multiple sizes for one pattern were not counted as separate models.

## Validation and database safety

See validation_report.md. The importer was run against a disposable in-memory SQLite database using the repository schema and importer function. A separate temporary parser fixture confirmed the importer's size parsing and exposed its 91/V defaults. The configured application database was not opened or modified, and no catalogue was imported into the live app.
