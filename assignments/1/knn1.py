import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('../../data/external/spotify.csv')

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['explicit'] = df['explicit'].astype(int)
    df['explicit']
    df.drop(['track_id','artists','album_name','track_name'], axis=1, inplace=True)
    df['track_genre'] = pd.Categorical(df['track_genre']).codes

    df.drop('Unnamed: 0', inplace=True, axis=1)

    #2.2.1
    
    correlation_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Audio Features')
    plt.savefig('figures/correlation_heatmap.png')
    # plt.show()

    df.drop(['loudness','acousticness'], axis=1, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_ms'], kde=True, bins=100)
    plt.title('Distribution of Track Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Count')
    plt.savefig('figures/duration_distribution.png')
    # plt.show()


    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['speechiness'], df['instrumentalness'], c=df['liveness'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Liveness')
    plt.title('Speechiness vs. Instrumentalness, colored by Liveness')
    plt.xlabel('Speechiness')
    plt.ylabel('Instrumentalness')
    plt.savefig('figures/speechiness_instrumentalness.png')
    # plt.show()


    df.to_csv('../../data/interim/spotify_modified.csv', index=False)

if __name__ == "__main__":
    main()