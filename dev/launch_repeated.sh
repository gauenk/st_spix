#!/bin/bash

while :
do
    # python ./dev/dev_prop_seg.py
    # python ./dev/prop_seg.py
    python ./dev/compare_prop_types.py
    if [[ "$?" -ne 0 ]]; then
      break
    fi
done
