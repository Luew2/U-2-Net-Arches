#!/bin/bash

# Create the .u2net directory if it doesn't exist
mkdir -p ~/.u2net

# Convert the saved model to ONNX format
echo "Converting the saved model to ONNX format..."
python model_converter.py

# Move the saved model to .u2net and overwrite the existing u2net.onnx
echo "Copying the ONNX model to ~/.u2net/u2net.onnx..."
cp u2net.onnx ~/.u2net/u2net.onnx -f

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Copy successful."
else
    echo "Copy failed."
fi
