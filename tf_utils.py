from os.path import basename, splitext
import pandas as pd
import numpy as np
import librosa
import ast
from numpy import array
from sklearn.preprocessing import LabelEncoder


def load_metadata(filepath):
    """
    Loads the fma data set's metadata to extract certain features
    :param filepath: the file path to the metadata file
    :return: a data frame containing the features contained inside the metadata file
    """
    filename = basename(filepath)

    # just read the csv as a data frame
    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    # formatting the data frame for tracks to contain key value pairs
    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        'category', categories=SUBSETS, ordered=True)

    COLUMNS = [('track', 'license'), ('artist', 'bio'),
               ('album', 'type'), ('album', 'information')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks


def extract(track_metadata, genre_metadata, path):
    """
    Extracts certain musical features from songs listed in path and obtains the genre from the songs
    using track_metadata and genre_metadata
    :param track_metadata: metadata file for tracks
    :param genre_metadata: metadata file for genres
    :param path: the file path to the songs of the fma data set
    :return: a list containing multiple features including genre for each song and a list of headers for those features
    """
    song_id = 0
    dataset = []
    mp3_files = librosa.util.find_files(path)  # load all mp3 files in the path, including inside subdirectories

    for i in xrange(len(mp3_files)):
        # for some reason, librosa's find_file returns two of each file in a path, and we don't want that
        if i % 2 == 0:
            features = []

            song_id = song_id + 1
            print("Reading Song#{}: ".format(song_id) + mp3_files[i] + "...")

            # Features related to music, such as beat, tempo, etc
            y, sr = librosa.load(mp3_files[i], duration=30)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rmse(y=y)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            genre = get_genre_for_current_song(track_metadata, genre_metadata, mp3_files[i])
            print(genre)

            # appending all the features to a list
            features.append(song_id)
            features.append(mp3_files[i])
            features.append(tempo)
            features.append(np.sum(beats))
            features.append(np.mean(chroma_stft))
            features.append(np.mean(rmse))
            features.append(np.mean(cent))
            features.append(np.mean(spec_bw))
            features.append(np.mean(rolloff))
            features.append(np.mean(zcr))
            for coefficient in mfcc:
                features.append(np.mean(coefficient))
            features.append(genre)

            # appending the list of features to another list for our dataset
            dataset.append(features)

    heading = ['id', 'songname', 'tempo', 'beats', 'chromagram', 'rmse',
           'centroid', 'bandwidth', 'rolloff', 'zcr', 'mfcc1', 'mfcc2',
           'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'genre']
		   
    return dataset, heading


def get_genre_for_current_song(track_metadata, genre_metadata, song):
    """
    Gets the musical genre for song
    :param track_metadata: metadata file for tracks
    :param genre_metadata: metadata file for genres
    :param song: the filename of the song
    :return: a string denoting the genre of the song
    """
    # Formatting file name to leave out the file path and extension, as well as remove leading 0's
    # eg. path\to\000002.mp3 becomes 2
    song_name = basename(song)
    song_name = splitext(song_name)[0]
    song_name = song_name.lstrip('0')
    song_no = int(song_name)

    # Checking to make sure that the genre_top value isn't nan
    if track_metadata.loc[song_no][("track", "genre_top")] == track_metadata.loc[song_no][("track", "genre_top")]:
        return track_metadata.loc[song_no][("track", "genre_top")]  # return the top_genre of the song
    # the top genre value was nan, so we find the song's most popular genre manually
    else:
        num_tracks = 0  # represents the number of tracks that belong to the genre
        genre_index = 0  # used to keep track of the index with the greatest number of tracks
        genre_list = track_metadata.loc[song_no][("track", "genres")]  # getting list of genres for the song

        # getting the most popular genre for the song
        for i in xrange(len(genre_list)):
            if genre_metadata.loc[genre_list[i]]["#tracks"] > num_tracks:
                num_tracks = genre_metadata.loc[genre_list[i]]["#tracks"]
                genre_index = i
        return genre_metadata.loc[genre_list[genre_index]]["title"]


def get_features(df, start, end):
    """
    returns df starting from start to end
    :param df: the df containing the features
    :param start: start
    :param end: end
    :return: data frame of specific features
    """
    return df.ix[:, start:end]


def get_labels(df, label):
    """
    returns df containing the labels
    :param df: the df containing the label
    :param label: label of the data set
    :return: df containing the label
    """
    return df[label]


def transform_to_matrix(df):
    """
    Transforms the data frame into a numpy array to use for training/testing
    :param df: the data frame to convert
    :return: a numpy array of features
    """
    return np.array(df)


def normalize(df):
    """
    Normalizes the data in the data frame
    :param df: the data frame with numerical features
    :return: a normalized data frame
    """
    features = list(df)
    for name in features:
        df[name] = (df[name]-np.min(df[name]))/(np.max(df[name])-np.min(df[name]))
    return df


def to_categorical(df):
    """
    Converts the data frame containing the labels into a numpy array of categorical labels
    :param df: the data frame containing the labels
    :return: a numpy array of categorical labels
    """
    values = array(df)
    label_encoder = LabelEncoder()
    values = label_encoder.fit_transform(values)
    return values
