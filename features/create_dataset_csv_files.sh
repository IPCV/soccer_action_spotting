#!/bin/bash

export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

# Ordered according SoccerNetV2 paper
declare -A labels_v2=(["Penalty"]="1" \
                      ["Kick-off"]="2" \
                      ["Goal"]="3" \
                      ["Substitution"]="4" \
                      ["Offside"]="5" \
                      ["Shots on target"]="6" \
                      ["Shots off target"]="7" \
                      ["Clearance"]="8" \
                      ["Ball out of play"]="9" \
                      ["Throw-in"]="10" \
                      ["Foul"]="11" \
                      ["Indirect free-kick"]="12" \
                      ["Direct free-kick"]="13" \
                      ["Corner"]="14" \
                      ["Yellow card"]="15" \
                      ["Red card"]="16"
                      ["Yellow->red card"]="17")

declare -A visibilities=( ["not shown"]="0" \
            ["visible"]="1" )

declare -A teams=( ["away"]="0" \
            ["home"]="1" \
            ["not applicable"]="2" )


function soccernet_pkg_path(){
  python - <<END
import os
import SoccerNet

soccernet_fpath = os.path.abspath(SoccerNet.__file__)
soccernet_pkg_path = os.path.dirname(soccernet_fpath)
print(soccernet_pkg_path)
END
}


function video_time_duration() {
  ffprobe -v error \
      -show_entries format=duration \
      -of default=noprint_wrappers=1:nokey=1 \
      "$1"
}


function make_videos_csv() {
  printf "match_path,match_date,visiting_team,home_team,score,first_half_duration_sec,second_half_duration_sec\n" > "$1"
  find "$dataset_path" -mindepth 4 -maxdepth 4 -type f -name 'Labels-v2.json' \
    | sort \
    | while read json_fpath; do
        match_path=`jq -r .UrlLocal "$json_fpath" `
        match_path=${match_path%/}
        match_date=`jq -r .gameDate "$json_fpath"`
        visiting_team=`jq -r .gameAwayTeam "$json_fpath"`
        home_team=`jq -r .gameHomeTeam "$json_fpath"`
        score=`jq -r .gameScore "$json_fpath"`

        first_half_duration_sec=`video_time_duration "$dataset_path"/"$match_path"/1.mkv`
        second_half_duration_sec=`video_time_duration "$dataset_path"/"$match_path"/2.mkv`

        printf "$match_path,$match_date,$visiting_team,$home_team,$score,$first_half_duration_sec,$second_half_duration_sec\n" >> "$1"
      done
}

function make_classes_csv(){
  printf "class\n" > "$1"
  printf "Background\n" >> "$1"
  printf "Penalty\n" >> "$1"
  printf "Kick-off\n" >> "$1"
  printf "Goal\n" >> "$1"
  printf "Substitution\n" >> "$1"
  printf "Offside\n" >> "$1"
  printf "Shots on target\n" >> "$1"
  printf "Shots off target\n" >> "$1"
  printf "Clearance\n" >> "$1"
  printf "Ball out of play\n" >> "$1"
  printf "Throw-in\n" >> "$1"
  printf "Foul\n" >> "$1"
  printf "Indirect free-kick\n" >> "$1"
  printf "Direct free-kick\n" >> "$1"
  printf "Corner\n" >> "$1"
  printf "Yellow card\n" >> "$1"
  printf "Red card\n" >> "$1"
  printf "Yellow->red card\n" >> "$1"
}

function parse_annotation() {
  game_time=`jq -r '.gameTime' <<< "$1"`
  label=${labels_v2[`jq -r '.label' <<< "$1"`]}
  position=`jq -r '.position' <<< "$1"`
  team=${teams[`jq -r '.team' <<< "$1"`]}
  visibility=${visibilities[`jq -r '.visibility' <<< "$1"`]}
  half=`echo "$game_time" | cut -c -1`
  game_time=`echo "$game_time" | cut -c 5-`
  printf "$match_path,$half,$game_time,$label,$position,$team,$visibility\n"
}


function make_split_csv() {
  printf "match_path,half,game_time,label,position,team,visibility\n" > "$2"
  for league in $(jq -r 'keys[]' "$1"); do
    for season in $(jq -r '."'"$league"'" | keys[]' "$1"); do
      readarray -t matches < <(jq -r '."'"$league"'" | ."'"$season"'" | .[]' "$1")
      for match in "${matches[@]}"; do
        export match_path="$league"/"$season"/"$match"
        labels_json="$dataset_path"/"$match_path"/Labels-v2.json
        jq -c '.annotations | .[]' "$labels_json" \
          | while read annotation; do
              parse_annotation "$annotation" >> "$2"
            done
      done
    done
  done
}


function make_challenge_csv() {
  printf "match_path\n" > "$2"
  for league in $(jq -r 'keys[]' "$1"); do
    for season in $(jq -r '."'"$league"'" | keys[]' "$1"); do
      readarray -t matches < <(jq -r '."'"$league"'" | ."'"$season"'" | .[]' "$1")
      for match in "${matches[@]}"; do
        match_path="$league"/"$season"/"$match"
        printf "$match_path\n" >> "$2"
      done
    done
  done
}


function make_splits_csv() {
  soccernet_splits_path=`soccernet_pkg_path`/data
  find "$soccernet_splits_path" -mindepth 1 -maxdepth 1 -type f -name 'SoccerNetGames*.json' \
    | while read split_json_fpath; do
        split_name=`echo "$split_json_fpath" \
                      | sed 's/.*SoccerNetGames//g' \
                      | sed 's/\.json//g' \
                      | awk '{print tolower($0)}'`
        split_csv_fpath="$1"/action_spotting_"$split_name"_v2.csv
        if [ "$split_name" != "challenge" ]; then
          make_split_csv "$split_json_fpath" "$split_csv_fpath"
        else
          make_challenge_csv "$split_json_fpath" "$split_csv_fpath"
        fi
      done
}


function usage() {
	echo "Usage: $0 [-d <dirpath>]" 1>&2;
	echo 1>&2;
	echo "  -o Sets CSV output dirpath" 1>&2;
  echo "  -v Only make video CSV" 1>&2;
  echo "  -s Only make splits CSV" 1>&2;
	echo "  -h This help message" 1>&2;
	exit 1;
}


function main(){

  export dataset_path=`realpath "$PROJECT_DIR"/data/soccernet`
  output_path="$dataset_path"

  classes=true
  videos=true
  splits=true
  while getopts ":o:cvsh" opt; do
    case "${opt}" in
      o) # Ouput dirpath
        output_path=`realpath ${OPTARG}`
        [ ! -d "$output_path" ] && usage
        ;;
      c) # Classes-only flag
        splits=''
        videos=''
        ;;
      v) # Videos-only flag
        classes=''
        splits=''
        ;;
      s) # Splits-only flag
        classes=''
        videos=''
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

  if [ $classes ]; then
    make_classes_csv "$output_path"/classes_v2.csv
  fi

  if [ $videos ]; then
    make_videos_csv "$output_path"/videos_v2.csv
  fi

  if [ $splits ]; then
    make_splits_csv "$output_path"
  fi
}

main "$@"
