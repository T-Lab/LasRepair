# Datasets

Dataset directories:

- `beers`
- `flight`
- `hospital`
- `movies`
- `rayyan`
- `shuttle`
- `tax_20k`
- `tax_200k`
- `walmart`

Their source, license, and redistribution status were not documented in the
original repository. Some tables also contained names, addresses, telephone
numbers, or author metadata. Verify the source license and privacy
requirements before adding any data to a public repository.

For an authorized dataset, use:

```text
datasets/<dataset_name>/clean.csv
datasets/<dataset_name>/dirty.csv
```

Optional controlled-error variants use:

```text
datasets/<dataset_name>/<dataset_name>_<error_percent>_error.csv
```
