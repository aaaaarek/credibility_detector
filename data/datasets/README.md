# Training Datasets

## Files

- `synthetic_articles.csv` contains generated MVP examples.
- `real_articles_seed.csv` is the hand-labeled real-article seed dataset. It starts empty on purpose.

## Real Article Schema

Required columns:

- `title`
- `content`
- `url`
- `source`
- `author`
- `publish_date`
- `source_links`
- `credibility_label`

Recommended columns:

- `label_reason`
- `dataset_source`

`credibility_label` is a float from `0.0` to `1.0`.

Suggested scale:

- `0.90-1.00`: official report, scientific article, or highly sourced news item.
- `0.70-0.85`: normal news item with author, date, and verifiable sources.
- `0.50-0.65`: opinion, blog, or partial context with some verification value.
- `0.30-0.45`: one-sided, weakly sourced, or clickbait-like text.
- `0.00-0.25`: scam, unsupported conspiracy, fabricated claim, or high-risk misinformation.

Use `|` to separate multiple source links in `source_links`.

Keep full copied article text only if you have the right to store it. Otherwise, store a short excerpt or notes allowed by your use case and keep the source URL for traceability.
