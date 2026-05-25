import pandas as pd
import logging
import time
from youtube_data_miner import extract_dislike_count

INPUT_CSV = 'youtube_game_videos.csv'
OUTPUT_CSV = 'youtube_game_videos_backfilled.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def main():
    df = pd.read_csv(INPUT_CSV)
    if 'video_id' not in df.columns or 'dislike_count' not in df.columns:
        logging.error('CSV must contain video_id and dislike_count columns.')
        return

    updated = 0
    for idx, row in df.iterrows():
        if pd.isna(row['dislike_count']) or str(row['dislike_count']).strip() == '':
            video_id = row['video_id']
            url = f'https://www.youtube.com/watch?v={video_id}'
            try:
                dislike_count = extract_dislike_count(url)
                df.at[idx, 'dislike_count'] = dislike_count
                updated += 1
                logging.info(f'Updated dislike_count for video {video_id}: {dislike_count}')
            except Exception as e:
                logging.error(f'Failed to get dislike_count for {video_id}: {e}')
            time.sleep(0.5)  # Be polite to the API

    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f'Backfill complete. {updated} rows updated. Output written to {OUTPUT_CSV}')

if __name__ == '__main__':
    main() 