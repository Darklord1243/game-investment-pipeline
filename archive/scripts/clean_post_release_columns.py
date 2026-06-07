image.pngimport pandas as pd

def clean_post_release_columns():
    """
    Clean the post-release dataset by removing unnamed columns and keeping only relevant columns
    """
    print("🧹 Cleaning post-release dataset columns...")
    
    # Load the post-release dataset
    post_release_df = pd.read_csv('post_release_dataset_final_clean.csv')
    
    print(f"📊 Original shape: {post_release_df.shape}")
    
    # Remove unnamed columns
    unnamed_cols = [col for col in post_release_df.columns if 'Unnamed' in col]
    print(f"🗑️  Removing {len(unnamed_cols)} unnamed columns")
    
    # Keep only relevant columns
    relevant_columns = [
        'game-name',
        'twitch_total_viewers', 'twitch_avg_viewers', 'twitch_max_viewers',
        'youtube_total_views', 'youtube_avg_views', 'youtube_max_views',
        'reddit_total_score', 'reddit_avg_score', 'reddit_max_score',
        'steam_current_players', 'steam_dlc_count', 'steam_has_achievements'
    ]
    
    # Filter to keep only relevant columns that exist
    existing_columns = [col for col in relevant_columns if col in post_release_df.columns]
    post_release_clean = post_release_df[existing_columns]
    
    print(f"✅ Cleaned shape: {post_release_clean.shape}")
    print(f"📋 Kept columns: {existing_columns}")
    
    # Save cleaned version
    post_release_clean.to_csv('post_release_dataset_clean_final.csv', index=False, encoding='utf-8')
    print("💾 Saved as: post_release_dataset_clean_final.csv")
    
    return post_release_clean

if __name__ == "__main__":
    clean_post_release_columns() 