#!/bin/bash

# This script is used to modify the input file and evaluate the output file

COLOR=""
POLICY=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while getopts "c:p:" opt; do
    case $opt in
        c)
            COLOR=$OPTARG
            ;;
        p)
            POLICY=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# Modify the input file
python "$SCRIPT_DIR/modify_png_colors.py" --color-mapping "$COLOR"

if [ $? -ne 0 ]; then
    echo "Script1 failed. Exiting."
    exit 1
fi

python "$SCRIPT_DIR/modified_eval.py" -f "$COLOR" --policy "$POLICY"

if [ $? -ne 0 ]; then
    echo "Script2 failed. Exiting."
    exit 1
fi