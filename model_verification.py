from datasets import load_from_disk 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch

# Check dataset
def dataset_check(dataset_path, num_sample_to_check):
    dataset = load_from_disk(dataset_path)
    for i in range(num_sample_to_check):
        print(f"Sample {i+1}:")
        print(dataset["train"][i]["text"])
        print("-"*80)

def model_check(prompt, model, tokenizer, device, max_length=128, temperature=1.0):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generated_ids = encoded_input["input_ids"]

    for i in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:,-1,:]
        logits = logits/temperature
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1).item()
        generated_ids = torch.cat((generated_ids, torch.tensor([[next_token_id]], device=device)), dim=1)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"input ==> {prompt}\n output ==> {output_text}")

if __name__ == "__main__":
    dataset_path = "D:\\SJZ\\tools\\LLM_dataset\\wikitext"
    dataset_check(dataset_path, 5)

    tokenizer = AutoTokenizer.from_pretrained(r"D:\SJZ\tools\nlp_models\openai_gpt2_tokenizer")
    model = AutoModelForCausalLM.from_pretrained(r"D:\\SJZ\\workspace_python\\LLM\\openai_gpt2_output\\final_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    model_check("Valkyria Chronicles III began development in 2010 , carrying", model, tokenizer, device)
