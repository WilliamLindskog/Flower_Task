#!/bin/bash

echo "Starting task..."

for lr in 0.001 0.01 0.1
do
    python -m task.main --config-path conf --config-name task learning_rate=$lr
done