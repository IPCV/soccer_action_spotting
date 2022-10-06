#!/bin/bash

export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"


function get_wav_fpath() {
  fname=$(basename "$1");
  dname=$(dirname "$1");
  fbname=${fname%.*};
  echo "$dname"/"$fbname".wav
}
export -f get_wav_fpath


function get_mp3_fpath() {
  fname=$(basename "$1");
  dname=$(dirname "$1");
  fbname=${fname%.*};
  echo "$dname"/"$fbname".mp3
}
export -f get_mp3_fpath


function to_wav() {
  overwrite="$1"
  wav_fpath="$(get_wav_fpath "$2")"
  if [ "$overwrite" == "-n" -a -f "$wav_fpath" ]; then
    return
  fi

  # Extracting raw audio from video
  tmp_raw_wav_fpath=$(mktemp /tmp/raw_XXXXXX.wav)
  ffmpeg -hide_banner -y -i "$2" -acodec pcm_s16le "$tmp_raw_wav_fpath"

  # Sampling at 16kHz, a bitrate of 90, monoaural for VGGish
  sox "$tmp_raw_wav_fpath" -C 90 -r 16k "$wav_fpath" remix 1,2
  rm "$tmp_raw_wav_fpath"
}
export -f to_wav


function usage() {
	echo "Usage: $0 [-d <dirpath>] [-k] [-n <int>]" 1>&2;
	echo 1>&2;
	echo "  -d Sets SoccerNet dataset dirpath" 1>&2;
	echo "  -k Keeps already processed WAV files" 1>&2;
	echo "  -n Number of cores to use for parallel processing" 1>&2;
	echo "  -h This help message" 1>&2;
	exit 1;
}

function main(){

  dataset_path=`realpath "$PROJECT_DIR"/data/soccernet`
  num_cores="$(nproc --ignore=1)"
  overwrite='-y'

  while getopts ":d:n:kh" opt; do
    case "${opt}" in
      d)  # SoccerNet dataset dirpath
        dataset_path=${OPTARG}
        ;;
      k)  # Keep already processed WAV files
        overwrite='-n'
        ;;
      n)  # Number of cores
        num_cores=$OPTARG
        ;;
      h)
        usage
        ;;
      *)
        usage
        ;;
    esac
  done
  shift $((OPTIND -1))

  find "$dataset_path" -type f -name '[12].mkv' \
    | parallel -P $num_cores -I {} to_wav "$overwrite" "{}"
}


main "$@"
