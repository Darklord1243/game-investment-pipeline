"""
Filter games from steam_all_games.csv based on:
- Release year 2010 or later
- More than 0 reviews
Saves result as steam_significant_games.csv
Games not meeting criteria are saved to steam_filtered_out_games.csv
"""
import pandas as pd
from typing import Optional
import re
from datetime import datetime


def extract_year(date_str: str) -> Optional[int]:
    """
    Extracts the year from various date string formats.
    Handles 'YYYY', 'YY', 'Month YYYY', 'Month DD, YYYY', and 'DD-Mon-YY'.
    Returns None if no valid year can be extracted.
    """
    if not isinstance(date_str, str):
        return None

    # Try parsing full dates first for accuracy
    for fmt in ('%d-%b-%y', '%b %d, %Y', '%Y-%m-%d', '%B %Y', '%b %Y'):
        try:
            return datetime.strptime(date_str, fmt).year
        except ValueError:
            continue

    # Fallback to regex for patterns like 'YYYY' or 'YY' at the end of a string
    # Match four-digit years directly
    match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if match:
        return int(match.group(1))

    # Match two-digit years and infer the century
    match = re.search(r'\b(\d{2})$', date_str)
    if match:
        year = int(match.group(1))
        # Handles 'YY' format, assuming '00-29' is 2000s and '30-99' is 1900s
        return 2000 + year if year < 30 else 1900 + year

    return None


def filter_games_by_year(input_csv: str, output_significant_csv: str, output_filtered_out_csv: str) -> None:
    """
    Filters games by release date and review count.
    - Saves games from 2010 or later with > 0 reviews to output_significant_csv.
    - Saves all other games to output_filtered_out_csv.
    """
    df = pd.read_csv(input_csv)

    # Convert 'review_count' to numeric, coercing errors to NaN
    df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0).astype(int)

    # Apply the year extraction function
    df['release_year'] = df['release_date'].apply(extract_year)

    # Define the filtering conditions
    significant_mask = (df['release_year'] >= 2010) & (df['review_count'] > 0)
    
    # Separate the dataframes
    significant_games_df = df[significant_mask].copy()
    filtered_out_games_df = df[~significant_mask].copy()

    # Drop the temporary 'release_year' column before saving
    significant_games_df.drop(columns=['release_year'], inplace=True)
    filtered_out_games_df.drop(columns=['release_year'], inplace=True)

    # Save the dataframes to their respective CSV files
    significant_games_df.to_csv(output_significant_csv, index=False)
    filtered_out_games_df.to_csv(output_filtered_out_csv, index=False)

    print(f"Saved {len(significant_games_df)} significant games to {output_significant_csv}")
    print(f"Saved {len(filtered_out_games_df)} filtered-out games to {output_filtered_out_csv}")


if __name__ == "__main__":
    filter_games_by_year(
        input_csv="steam_all_games.csv",
        output_significant_csv="steam_significant_games.csv",
        output_filtered_out_csv="steam_filtered_out_games.csv"
    ) 