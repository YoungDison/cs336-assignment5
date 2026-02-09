import pandas as pd
from torch.utils.data import Dataset
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
from unittest.mock import patch



def statistics(df_path):
    df = pd.read_parquet(df_path)
    groups = df.groupby(['format_reward', 'answer_reward'])
    split_groups = {name: group for name, group in groups}

    reward0_0, reward1_0, reward1_1 = split_groups[(0.0,0.0)], split_groups[(1.0,0.0)], split_groups[(1.0,1.0)]
    reward0_1 = split_groups.get((0.0,1.0), pd.DataFrame())
    stats = {"total cases":len(df), "only_format_right_rate":len(reward1_0)/len(df),
             "both_right_rate":len(reward1_1)/len(df),\
                "both_wrong_rate":len(reward0_0)/len(df)}
    print(stats)

def preprocess_data(data_path, outpath):
    df = pd.read_parquet(data_path)
    df["think"] = [answer.split("#### ")[0] for answer in df["answer"]]
    df["answer"] = [answer.split("#### ")[1] for answer in df["answer"]]
    df.to_parquet(outpath)

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )

def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.drive_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def tokenize_prompt_and_output(prompt_strs: list[str], \
                               output_strs: list[str], \
                                tokenizer):
    prompt_tokens = tokenizer(prompt_strs)["input_ids"]
    output_tokens = tokenizer(output_strs)["input_ids"]
    length = -1
    for prompt, output in zip(prompt_tokens, output_tokens):
        length = max(length, len(prompt)+len(output)-1)
    padding = tokenizer.pad_token_id
    input_ids_list, labels_list, mask_list = [], [], []
    for prompt, output in zip(prompt_tokens, output_tokens):
        mask = [0] * length
        total_token = prompt + output
        now_len = len(total_token)
        mask[len(prompt)-1:now_len-1] = [1] * (len(output))
        total_token += [padding] * (length + 1 - now_len)
        input_ids, labels = total_token[:-1], total_token[1:]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        mask_list.append(mask)
    input_tensor = torch.tensor(input_ids_list)
    label_tensor = torch.tensor(labels_list)
    mask_tensor = torch.tensor(mask_list)
    return {"input_ids": input_tensor, \
            "labels": label_tensor, \
            "response_mask": mask_tensor}

class SFTDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.df = pd.read_parquet(data_path)
    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()
    # return a dict like {"prompt":..., "output":...}
    def __len__(self):
        return len(self.df)

def collate_fn_sft(batch: list[dict], tokenizer):
    prompts = [data["prompt"] for data in batch]
    outputs = [data["output"] for data in batch]
    tokenized_batch = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return tokenized_batch     

if __name__ == "__main__":
    data_path = "/root/shared-nvme/gsm8k/main/test-00000-of-00001.parquet"
    outpath = "/root/shared-nvme/gsm8k/main/text-preprocessed.parquet"
    preprocess_data(data_path, outpath)