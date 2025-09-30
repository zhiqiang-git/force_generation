#!/bin/bash

LOCAL_ROOT="/home/zhiqiang/Documents/Physical-stability/main/python/data/chair_1200"
REMOTE_USER=zhiqiang
REMOTE_HOST=UT_A5000
REMOTE_ROOT="/home/zhiqiang/Documents/phy/python/data/chair_600"

for local_file in $(find "$LOCAL_ROOT" -type f -name "data.pkl"); do
    shape_id=$(basename "$(dirname "$(dirname "$local_file")")")
    remote_dir="$REMOTE_ROOT/$shape_id"

    echo "Transferring $local_file → $REMOTE_HOST:$remote_dir/data.pkl"

    ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p '$remote_dir'"
    scp "$local_file" $REMOTE_USER@$REMOTE_HOST:"$remote_dir/data.pkl"
done
