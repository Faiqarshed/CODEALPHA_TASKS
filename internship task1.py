import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set up Spotify API credentials
client_id = 'b3ca158bb1a44bcd84a12ac3bb9284ad'
client_secret = '8468d8136a214e2295045cc29ca4c937'

# Create a Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



# Define a function to collect user-song interactions
def collect_user_song_interactions(user_id):
    # Get the user's playlists
    playlists = sp.user_playlists(user_id)

    # Initialize a list to store user-song interactions
    interactions = []

    # Iterate over each playlist
    for playlist in playlists['items']:
        # Get the playlist tracks
        tracks = sp.playlist_tracks(playlist['id'])

        # Iterate over each track
        for track in tracks['items']:
            # Extract the song ID and user ID
            song_id = track['track']['id']
            user_id = user_id

            # Add the interaction to the list
            interactions.append((user_id, song_id))

    return interactions


# Collect user-song interactions for a sample user
user_id = '1234567890'
interactions = collect_user_song_interactions(user_id)

# Convert the interactions to a Pandas DataFrame
df = pd.DataFrame(interactions, columns=['user_id', 'song_id'])

# Create a user-song interaction matrix using groupby
user_song_matrix = df.groupby(['user_id', 'song_id']).size().unstack(fill_value=0)

# Split the data into training and testing sets
X = user_song_matrix
y = user_song_matrix['repeated_play']  # Note: this column doesn't exist in the original code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))