# double checking data
# import pandas as pd

# df = pd.read_csv("data/raw/nyt_google_enriched_partial.csv")

# print(f"Total books enriched so far: {len(df)}")
# print(f"Columns: {df.columns.tolist()}")
# print(f"\nMissing values:\n{df.isnull().sum()}")
# print(f"\nSample:\n{df.head()}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path

# Create folder for figures
Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/raw/nyt_google_enriched.csv")

# Safely extract the first category/genre
def get_primary_genre(x):
    if pd.isna(x) or x == "[]":
        return "Unknown"
    
    try:
        categories = ast.literal_eval(x)
        if len(categories) > 0:
            return categories[0]
        else:
            return "Unknown"
    except:
        return "Unknown"

# categories comes back as a list — extract the first one
df["primary_genre"] = df["categories"].apply(get_primary_genre)

# Graph 1: Top Genres 

genre_counts = df['primary_genre'].value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='Greens_r')
plt.title('Top 15 genres among NYT bestsellers')
plt.xlabel('Number of books')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('figures/eda_genre_distribution.png', dpi=150)
plt.close()

# Graph 2: Weeks on list distribution 

plt.figure(figsize=(10, 5))
sns.histplot(df["nyt_weeks_on_list"].dropna(), bins=50, color="#639922")

median_weeks = df["nyt_weeks_on_list"].median()

plt.axvline(
    median_weeks,
    color="red",
    linestyle="--",
    label=f"Median: {median_weeks} weeks"
)

plt.title('Distribution of weeks on NYT bestseller list')
plt.xlabel('Weeks on list')
plt.ylabel('Number of books')
plt.legend()
plt.tight_layout()
plt.savefig('figures/eda_weeks_distribution.png', dpi=150)
plt.close()

# Graph 3: Publication month by genre heatmap 
# extract month from published date
df['pub_month'] = pd.to_datetime(
    df['google_published_date'], errors='coerce'
).dt.month

month_genre = df.groupby(['pub_month', 'primary_genre']).size().unstack(fill_value=0)
top_genres = df['primary_genre'].value_counts().head(6).index
month_genre = month_genre[top_genres]

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(12, 6))
sns.heatmap(
    month_genre,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=top_genres,
    yticklabels=[month_names[int(month) - 1] for month in month_genre.index]
)

plt.title('Bestseller debuts by publication month and genre')
plt.xlabel('Genre')
plt.ylabel('Month')
plt.tight_layout()
plt.savefig('figures/eda_month_heatmap.png', dpi=150)
plt.close()

print("EDA complete. Saved 3 figures:")
print("figures/eda_genre_distribution.png")
print("figures/eda_weeks_distribution.png")
print("figures/eda_month_heatmap.png")