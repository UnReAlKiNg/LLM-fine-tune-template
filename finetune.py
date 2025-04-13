from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass; expect that inputs contains "labels"
        outputs = model(**inputs)
        loss = outputs.get("loss")
        if loss is None:
            # Manually compute the loss if not provided.
            # For causal LM, shift the logits and labels
            logits = outputs.get("logits")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return (loss, outputs) if return_outputs else loss

def NormalTraining(model_path, tokenized_dataset):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # model = nn.DataParallel(model.cuda(), device_ids=[0, 1])

    training_args = TrainingArguments(
        # output_dir = "./gpt2_chinese_output",
        output_dir = "./openai_gpt2_output",
        eval_strategy = "steps",
        eval_steps = 500,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 4,
        num_train_epochs = 8,
        weight_decay = 0.01,
        logging_steps = 100,
        remove_unused_columns=False,
        save_steps=1000
    )
    # trainer = Trainer(
    #     model = model,
    #     args = training_args,
    #     train_dataset = tokenized_dataset["train"],
    #     eval_dataset = tokenized_dataset["validation"]
    # )
    trainer = MyTrainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["validation"]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    trainer.save_model("./openai_gpt2_output/final_model")

def PPOTraining(model_path, tokenizer, tokenized_dataset):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        output_dir="./openai_gpt2_output/PPOTraining"
    )
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=None,
        reward_model=None,
        processing_class=None,
        args=ppo_config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    def reward_function(texts):
        return [torch.tensor(len(text)) for text in texts]

    for epoch in range(ppo_config.num_ppo_epochs):
        for batch in ppo_trainer.dataloader:
            queries = batch["query"]

            # 生成响应
            response_tensors = []
            for query in queries:
                inputs = tokenizer(query, return_tensors="pt")
                generation = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9
                )
                response_tensors.append(generation[0])

            batch["response"] = [tokenizer.decode(r) for r in response_tensors]

            # 计算奖励
            rewards = reward_function(batch["response"])

            # PPO训练步骤
            stats = ppo_trainer.step(response_tensors, rewards)
            print(f"Step stats: {stats}")



if __name__ == "__main__":
    dataset = load_from_disk("D:\\SJZ\\tools\\LLM_dataset\\wikitext")
    # tokenizer = AutoTokenizer.from_pretrained("D:\\SJZ\\tools\\nlp_models\\sentence-transformer-all--MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\SJZ\tools\nlp_models\openai_gpt2_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    NormalTraining(r"D:\SJZ\tools\nlp_models\openai_gpt2_model", tokenized_dataset)
    # PPOTraining(r"D:\SJZ\tools\nlp_models\openai_gpt2",tokenizer, tokenized_dataset)
    # PPOTraining(r"D:\SJZ\tools\nlp_models\gpt2_chinese_model",tokenizer, tokenized_dataset)
