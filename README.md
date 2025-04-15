# LLM-fine-tune-template
A template that can quickly have a try on fine-tuning a LLM

`Model` and corresponding `tokenizer` i downloaded from Huggingface and saved to local. You can simply changed the location to yours and click run to fine-tune the pre-trained model. Note that the PPO-part in `finttune.py` currently not working, as the `trl` library (0.16.1) has many difference to (0.7.1). I will complete this part in future.

If you want to try it on your own data (suppose you have put those sentenses into a `.txt` file), then you can use the struct in `CustomizedDataset.py` to build your own dataset. Note that, in this case, you can 
1. delete `["train"]` and `["validation"]`
2. use `train_test_split` function in sklearn and then put in "train" set and "test" set separately.
