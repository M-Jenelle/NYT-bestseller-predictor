# EDA Feature Design Summary

## Goal

This analysis combines the NYT + Google Books dataset with the Open Library enrichment dataset to understand which fields are useful for feature design before modeling bestseller performance.

The main modeling target explored here is `nyt_weeks_on_list`. The script also creates helper target flags:

- `long_run_26_weeks`: book stayed on the NYT list for at least 26 weeks
- `long_run_52_weeks`: book stayed on the NYT list for at least 52 weeks
- `above_median_weeks`: book stayed on the list at least as long as the median book

## Dataset Overview

| dataset               |   rows |   columns |   unique_isbn13 |   duplicate_isbn13_rows |
|:----------------------|-------:|----------:|----------------:|------------------------:|
| nyt_google_enriched   |   1811 |        41 |            1790 |                      20 |
| open_library_enriched |   1054 |        28 |            1054 |                       0 |
| merged_feature_design |   1811 |        76 |            1790 |                      20 |

## Key Data Quality Notes

- Merged dataset rows: **1,811**
- Rows with an Open Library match: **1,054** (58.2%)
- Median NYT weeks on list: **2**
- Share with 26+ weeks: **12.3%**
- Share with 52+ weeks: **6.8%**

## Top Missingness Issues

| column                         |   missing_percent |
|:-------------------------------|------------------:|
| ratings_count                  |             95.58 |
| average_rating                 |             95.58 |
| series_name                    |             94.48 |
| author_birth_date              |             79.57 |
| page_count_gap_google_minus_ol |             55.6  |
| ol_number_of_pages             |             55.55 |
| google_subtitle                |             52.18 |
| ol_author_name                 |             42.9  |
| google_vs_ol_year_gap          |             42.52 |
| ol_first_publish_year          |             42.41 |
| years_since_first_publish      |             42.41 |
| ol_first_publish_decade        |             42.41 |
| ol_publisher                   |             42.3  |
| ol_ebook_access                |             41.86 |
| ol_edition_count               |             41.86 |

Use fields with very high missingness carefully. For example, `average_rating` and `ratings_count` may look useful, but if most rows are missing them, they can create bias or reduce model coverage.

## Top Genres

| primary_genre             |   count |
|:--------------------------|--------:|
| Fiction                   |     677 |
| Juvenile Fiction          |     248 |
| Biography & Autobiography |     209 |
| Young Adult Fiction       |     148 |
| Business & Economics      |      52 |
| History                   |      44 |
| Comics & Graphic Novels   |      43 |
| Political Science         |      41 |
| Cooking                   |      40 |
| Self-Help                 |      35 |

## Top NYT Lists

| nyt_list_name                      |   count |
|:-----------------------------------|--------:|
| Hardcover Fiction                  |     221 |
| Advice, How-To & Miscellaneous     |     165 |
| Hardcover Nonfiction               |     162 |
| Combined Print & E-Book Fiction    |     132 |
| Mass Market                        |     130 |
| Paperback Trade Fiction            |     127 |
| Children’s Picture Books           |     100 |
| Young Adult Hardcover              |      93 |
| Combined Print & E-Book Nonfiction |      85 |
| Paperback Nonfiction               |      80 |

## Numeric Features Most Correlated with NYT Weeks on List

This is an early screening step, not proof of causation.

| feature                   |   correlation_with_nyt_weeks_on_list |
|:--------------------------|-------------------------------------:|
| ol_edition_count          |                            0.534346  |
| num_archive_ids           |                            0.491509  |
| num_ol_subjects           |                            0.320132  |
| google_pub_year           |                           -0.291443  |
| years_since_google_pub    |                            0.291443  |
| ol_first_publish_year     |                           -0.24486   |
| years_since_first_publish |                            0.24486   |
| num_ol_languages          |                            0.193962  |
| google_vs_ol_year_gap     |                            0.159362  |
| average_rating            |                           -0.112833  |
| nyt_rank                  |                           -0.0842224 |
| author_total_works        |                            0.0809425 |
| ratings_count             |                           -0.0743531 |
| description_word_count    |                           -0.057774  |
| num_lists_appeared        |                           -0.0421991 |

## Candidate Modeling Features

| feature                           | source                      | feature_type         | why_it_may_help                                                                                                     |
|:----------------------------------|:----------------------------|:---------------------|:--------------------------------------------------------------------------------------------------------------------|
| primary_genre                     | Google Books                | categorical          | Genre/category may strongly affect bestseller shelf life and target audience.                                       |
| nyt_list_name                     | NYT                         | categorical          | Different NYT lists behave differently; children's books, fiction, and nonfiction may have different staying power. |
| page_count / ol_number_of_pages   | Google Books + Open Library | numeric              | Length may separate children's books, novels, nonfiction, and reference-style books.                                |
| google_pub_month                  | Google Books                | date/seasonality     | Publication timing may matter because some genres are seasonal.                                                     |
| description_word_count            | Google Books                | text-derived numeric | Richer metadata could correlate with better match quality or more commercially supported releases.                  |
| average_rating / ratings_count    | Google Books                | numeric              | Reader engagement could help, but missingness is very high so these should be used carefully.                       |
| is_series / series_name           | Open Library                | categorical/boolean  | Series books may have built-in audiences and repeat demand.                                                         |
| author_total_works                | Open Library                | numeric              | Author productivity/popularity proxy; prolific authors may have more existing audience awareness.                   |
| ol_edition_count                  | Open Library                | numeric              | More editions can proxy for popularity, longevity, translations, or reprints.                                       |
| ol_ebook_access / ol_has_fulltext | Open Library                | categorical/boolean  | Availability/access fields may signal older/public/archived books versus newer commercial releases.                 |

## Bestseller vs Non-Bestseller Class EDA

The negative samples from `data/raw/negative_samples.csv` are aligned with the NYT positives on shared fields before comparing coverage and distributions.

Class balance:

| class                   |   count |   percent |
|:------------------------|--------:|----------:|
| negative_non_bestseller |    5493 |     75.21 |
| positive_bestseller     |    1811 |     24.79 |

- Positive NYT bestseller rows: **1,811**
- Negative non-bestseller rows: **5,493**
- Negative-to-positive ratio: **3.03:1**

Feature coverage comparison:

| feature               |   negative_non_bestseller |   positive_bestseller |
|:----------------------|--------------------------:|----------------------:|
| author                |                     99.65 |                 98.23 |
| isbn13_clean          |                    100    |                 98.84 |
| ol_ebook_access       |                    100    |                 58.14 |
| ol_edition_count      |                    100    |                 58.14 |
| ol_first_publish_year |                    100    |                 57.59 |
| ol_languages          |                    100    |                 52.68 |
| ol_subjects           |                     46.71 |                 28.71 |
| page_count            |                     51.77 |                 99.83 |
| publish_year          |                    100    |                 99.67 |
| publisher             |                     99.89 |                 99.67 |
| title                 |                    100    |                100    |

Shared numeric feature distribution summary:

| class_label             | feature          |   count |       mean |       std |   min |     25% |   50% |   75% |   max |
|:------------------------|:-----------------|--------:|-----------:|----------:|------:|--------:|------:|------:|------:|
| negative_non_bestseller | page_count       |    2844 |  248.275   | 220.511   |     1 |  126.75 |   230 |   320 |  6394 |
| negative_non_bestseller | ol_edition_count |    5493 |    1.29583 |   0.70301 |     1 |    1    |     1 |     1 |    10 |
| negative_non_bestseller | publish_year     |    5493 | 2024.01    |   3.78012 |  1980 | 2025    |  2025 |  2025 |  2026 |
| positive_bestseller     | page_count       |    1808 |  179.267   | 217.452   |     0 |    0    |     0 |   337 |  1679 |
| positive_bestseller     | ol_edition_count |    1053 |    7.05318 |  24.7657  |     1 |    1    |     2 |     5 |   398 |
| positive_bestseller     | publish_year     |    1805 | 2024.01    |   3.79191 |  1980 | 2025    |  2025 |  2025 |  2027 |

## Generated Figures

- `figures/eda_01_missingness_top20.png`
- `figures/eda_02_source_overlap.png`
- `figures/eda_03_target_weeks_distribution.png`
- `figures/eda_04_top_nyt_lists.png`
- `figures/eda_05_top_google_genres.png`
- `figures/eda_06_pub_month_genre_heatmap.png`
- `figures/eda_07_numeric_feature_correlations.png`
- `figures/eda_08_page_count_source_comparison.png`
- `figures/eda_09_series_vs_weeks_boxplot.png`
- `figures/eda_10_author_works_vs_weeks.png`
- `figures/eda_11_long_run_rate_by_genre.png`
- `figures/eda_12_edition_count_vs_weeks.png`
- `figures/eda_13_class_balance.png`
- `figures/eda_14_feature_coverage_by_class.png`
- `figures/eda_15_shared_feature_distributions_by_class.png`

## Next Modeling Direction

A good first modeling target is either:

1. Regression: predict `nyt_weeks_on_list`
2. Classification: predict `long_run_52_weeks` or `long_run_26_weeks`

For a first pass, start with interpretable features like genre, NYT list name, page count, publication month, description length, series flag, author total works, and Open Library edition count. Then compare a simple baseline model against tree-based models.
