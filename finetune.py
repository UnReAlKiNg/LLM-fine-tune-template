from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from trl import PPOConfig, PPOTrainer, create_reference_model
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model

# set the GPU that you want to use
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Full_FineTune:
    """
    Train the model from 0.

    For initialization, you need to provide model_path, tokenizer_path and dataset_path.

    Model and tokenizer can be downloaded from Hugging-Face, and dataset can be downloaded from Datasets library
    """

    def __init__(self, model_path, tokenizer_path, dataset_path, batched=True):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Full-Fine-Tune >> model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Full-Fine-Tune >> tokenizer loaded")

        self.dataset = load_from_disk(dataset_path)
        print("Full-Fine-Tune >> dataset loaded")

        print("Full-Fine-Tune >> Start mapping dataset")
        self.tokenized_dataset = self.dataset.map(self.tokenize_function_normal, batched=batched)
        print("Full-Fine-Tune >> dataset mapping done")

        self.training_args = TrainingArguments()
        self.Trainer = Trainer(self.model)
        self.eval_results = None

    def set_Training_args(self,
                          output_dir="./Full_FineTune_Output",
                          eval_strategy="steps",
                          eval_steps=500,
                          per_device_train_batch_size=8,
                          per_device_eval_batch_size=8,
                          num_train_epochs=100,
                          weight_decay=0.01,
                          logging_steps=100,
                          remove_unused_columns=False,
                          save_steps=10000
                          ):
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            remove_unused_columns=remove_unused_columns,
            save_steps=save_steps
        )

    def tokenize_function_normal(self, examples):
        tokenized_output = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    def train(self, model_save_path):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"]
            # train_dataset=tokenized_dataset,
            # eval_dataset=tokenized_dataset
        )

        print("Full-Fine-Tune >> start training model")
        self.trainer.train()
        print("Full-Fine-Tune >> finish training model")

        self.eval_results = self.trainer.evaluate()
        print(f"Full-Fine-Tune >> evaluation results: {self.eval_results}")

        self.trainer.save_model(model_save_path)
        print(f"Full-Fine-Tune >> trained model saved to {model_save_path}")


class LoRA_FineTune:
    """
    Implement for LoRA fine-tune.

    Note that, you should install peft library (pip install peft) before using.
    """
    def __init__(self, model_path, tokenizer_path, dataset_path, batched=True):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        print("LoRA-Fine-Tune >> model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("LoRA-Fine-Tune >> tokenizer loaded")

        self.dataset = load_from_disk(dataset_path)
        print("LoRA-Fine-Tune >> dataset loaded")

        print("LoRA-Fine-Tune >> Start mapping dataset")
        self.tokenized_dataset = self.dataset.map(self.tokenize_function_normal, batched=batched)
        print("LoRA-Fine-Tune >> dataset mapping done")

        self.peft_config = LoraConfig()
        self.training_args = TrainingArguments()
        self.Trainer = Trainer(self.model)
        self.eval_results = None

    def set_peft_model(self,
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=16,
                        lora_alpha=8,
                        lora_dropout=0.1
                       ):
        self.peft_config = LoraConfig(
                            task_type=task_type,
                            inference_mode=inference_mode,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout
                        )
        self.peft_model = get_peft_model(self.model, self.peft_config)
        print(f"LoRA-Fine-Tune >> peft_model done, info: {self.peft_model.print_trainable_parameters()}")


    def set_Training_args(self,
                          output_dir="./LoRA_FineTune_Output",
                          eval_strategy="steps",
                          eval_steps=500,
                          per_device_train_batch_size=8,
                          per_device_eval_batch_size=8,
                          num_train_epochs=100,
                          weight_decay=0.01,
                          logging_steps=100,
                          remove_unused_columns=False,
                          save_steps=10000
                          ):
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            remove_unused_columns=remove_unused_columns,
            save_steps=save_steps
        )

    def tokenize_function_normal(self, examples):
        tokenized_output = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    def train(self, model_save_path):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"]
            # train_dataset=tokenized_dataset,
            # eval_dataset=tokenized_dataset
        )

        print("LoRA-Fine-Tune >> start training model")
        self.trainer.train()
        print("LoRA-Fine-Tune >> finish training model")

        self.eval_results = self.trainer.evaluate()
        print(f"LoRA-Fine-Tune >> evaluation results: {self.eval_results}")

        self.trainer.save_model(model_save_path)
        print(f"LoRA-Fine-Tune >> trained model saved to {model_save_path}")


class PPO_FineTune:
    def __init__(self, model_path, tokenizer_path, dataset_path, batched=True):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        print("PPO-Fine-Tune >> model loaded")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        print("PPO-Fine-Tune >> reward_model loaded")
        self.ref_model = create_reference_model(self.model)
        print("PPO-Fine-Tune >> ref_model loaded")
        self.value_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        print("PPO-Fine-Tune >> value_model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("PPO-Fine-Tune >> tokenizer loaded")

        self.dataset = load_from_disk(dataset_path)
        print("PPO-Fine-Tune >> dataset loaded")

        print("PPO-Fine-Tune >> Start mapping dataset")
        self.tokenized_dataset = self.dataset.map(self.tokenize_function_normal, batched=batched)
        print("PPO-Fine-Tune >> dataset mapping done")

        self.training_args = TrainingArguments()
        self.Trainer = PPOConfig()
        self.eval_results = None

    def set_Training_args(self,
                          output_dir="./LoRA_FineTune_Output",
                          learning_rate=1.41e-5,
                          mini_batch_size=2,
                          batch_size=4,
                          num_ppo_epochs=1,
                          gradient_accumulation_steps=1,
                          ):
        self.training_args = PPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            batch_size=batch_size,
            num_ppo_epochs=num_ppo_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

    def tokenize_function_normal(self, examples):
        tokenized_output = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    def train(self, model_save_path):
        self.trainer = PPOTrainer(
            args=self.training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"]
            # train_dataset=tokenized_dataset,
            # eval_dataset=tokenized_dataset
        )

        print("PPO-Fine-Tune >> start training model")
        self.trainer.train()
        print("PPO-Fine-Tune >> finish training model")

        self.eval_results = self.trainer.evaluate()
        print(f"PPO-Fine-Tune >> evaluation results: {self.eval_results}")

        self.trainer.save_model(model_save_path)
        print(f"PPO-Fine-Tune >> trained model saved to {model_save_path}")



if __name__ == "__main__":
    # common used params
    tokenizer_path = r"D:\SJZ\tools\nlp_models\openai_gpt2_tokenizer"
    model_path = r"D:\SJZ\tools\nlp_models\openai_gpt2_model"
    dataset_path = "D:\\SJZ\\tools\\LLM_dataset\\wikitext"

    # Full fine tune test --------------------
    # FFT = Full_FineTune(model_path=model_path, tokenizer_path=tokenizer_path, dataset_path=dataset_path)
    # FFT.set_Training_args()
    # FFT.train(model_save_path="./openai_gpt2_output/FullFineTune_final_model")
    # Full fine tune test --------------------

    # LoRA fine tune test --------------------
    # LoRAFT = LoRA_FineTune(model_path=model_path, tokenizer_path=tokenizer_path, dataset_path=dataset_path)
    # LoRAFT.set_Training_args()
    # LoRAFT.train(model_save_path="./openai_gpt2_output/LoRAFineTune_final_model")
    # LoRA fine tune test --------------------

    # PPO fine tune test --------------------
    PPOFT = PPO_FineTune(model_path=model_path, tokenizer_path=tokenizer_path, dataset_path=dataset_path)
    PPOFT.set_Training_args()
    PPOFT.train(model_save_path="./openai_gpt2_output/LoRAFineTune_final_model")
    # PPO fine tune test --------------------
