from vllm import LLM, SamplingParams
import pandas as pd
from typing import List, Callable
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_llm(vllm_model: LLM,
                 reward_fn: Callable[[str, str], dict[str, float]],
                 datapath,
                 eval_sampling_params: SamplingParams,
                 output_path: str = "math_baseline_results.parquet"):
    ori_df = pd.read_parquet(datapath)
    prompts = [f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>""" for question in ori_df["question"]]
    llm_outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_text = []
    format_reward, answer_reward, reward = [], [], []
    for i in range(len(llm_outputs)):
        text = llm_outputs[i].outputs[0].text
        generated_text.append(text)
        metrics = reward_fn(text, ori_df["answer"].iloc[i])
        format_reward.append(metrics["format_reward"])
        answer_reward.append(metrics["answer_reward"])
        reward.append(metrics["reward"])
    df = pd.DataFrame()
    df["prompt"] = prompts
    df["generated_text"] = generated_text
    df["format_reward"] = format_reward
    df["answer_reward"] = answer_reward
    df["reward"] = reward
    df["ground_truth"] = ori_df["answer"]
    df.to_parquet(output_path)
    print(f"Results serialized to {output_path}!")

if __name__=="__main__":
    model = LLM(model="/root/shared-nvme/Qwen2.5")
    sampling_params = SamplingParams(temperature=1.0, 
                                     top_p=1.0, max_tokens=1024,
                                     stop=["</answer>"],
                                     include_stop_str_in_output=True)
    datapath = "/root/shared-nvme/gsm8k/main/test-preprocessed.parquet"
    reward_fn = r1_zero_reward_fn
    evaluate_llm(model, reward_fn, datapath, sampling_params)



    
    