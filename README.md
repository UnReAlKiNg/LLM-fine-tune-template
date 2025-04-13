# LLM-fine-tune-template
A template that can quickly have a try on fine-tuning a LLM

`Model` and corresponding `tokenizer` i downloaded from Huggingface and saved to local. You can simply changed the location to yours and click run to fine-tune the pre-trained model. Note that the PPO-part in `finttune.py` currently not working, as the `trl` library (0.16.1) has many difference to (0.7.1). I will complete this part in future.
