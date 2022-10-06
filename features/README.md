# Feature Extraction from SoccerNet


## CSV splits creation

The following command converts SoccerNet JSON annotations to CSV files. It creates one CSV file describing the videos and one CSV file for each action spotting split.

```shell
./create_dataset_csv_files.sh
```

## Audio

### Extracting the audio from all videos

By default, the following command extracts WAV audio files from all the low qualities videos.\*

```shell
./extract_soccernet_wav.sh
```

\**Other options are available typing the -h argument option.*

### Calculating features files from WAV audio files

The next command calculates the VGGish feature vectors from all WAV files and stores them in the same directory as the WAV files. 

```shell
python features/extract_vggish_features.py
```
