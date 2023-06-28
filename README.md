# Anonymous submission

## Training

1. Placing `arxiv-metadata-oai-snapshot.json` and `scicap_data` into project folder.
2. Applying OCR, see example in `example\`
2. Converting dataset into MAGMA format `convert_dataset.py`
3. Training via `deepspeed train.py --config path_to_my_config`

## Inferencing

See `inference_evaluate/inference_image_tokens.py`

## Inference checkpoints

See `checkpoints/*/test_output.pt*`