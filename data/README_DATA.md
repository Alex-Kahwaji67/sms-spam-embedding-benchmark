# Data Notes

Raw SMS data is intentionally not included in this staged repo.

The coursework notebook used a file named:

```text
spam.csv
```

with:

- `v1`: label (`ham` or `spam`)
- `v2`: SMS message text

To run the notebook, place the file here:

```text
data/spam.csv
```

The original notebook loaded the file with CP1252 encoding:

```python
pd.read_csv("spam.csv", encoding="cp1252")
```

Before publishing any dataset copy, verify that redistribution is allowed. The raw SMS text includes phone numbers, URLs, adult/spam content, and conversational personal messages, so the safer portfolio default is to provide download instructions rather than commit the data.

