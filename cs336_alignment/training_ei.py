from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import utils
from functools import partial
from cs336_alignment import sft, drgrpo_grader, baseline
from vllm import SamplingParams
import pandas as pd
import os
import wandb
import random



ModelPath = "/root/shared-nvme/sft_checkpoints/Epoch64Uniquesamples1024/"
DatasetTrain = "/root/shared-nvme/gsm8k/main/train-prompt_and_answer.parquet"
DatasetTest = "/root/shared-nvme/gsm8k/main/test-prompt_and_answer.parquet"
OutPutDir = "/root/shared-nvme/ei_checkpoints"
FinalTestPath = "/root/assignment5-alignment/Comparasion-ei"
LearningRate = 1e-5
Epoch = 64
NEiSteps = 5
MicroBatchSize = 2


GradientAccumulationSteps = 8
BatchSize = 512
Rollout = 4
UniqueSamples = 7144
OurPutPath = os.path.join(OutPutDir, f"UniqueSample{UniqueSamples}.parquet")
TrainingDevice = torch.device("cuda:0")
InferenceDevice = torch.device("cuda:1")
InferenceSeed = 42
InferenceGpuMemoryUti = 0.75
Seed = 42
SamplingMinTokens = 4
G = 4
SamplingParameters = SamplingParams(temperature=1.0, 
                                     top_p=1.0, 
                                     max_tokens=1024,
                                     min_tokens=SamplingMinTokens,
                                     n=G,
                                     stop=["</answer>"],
                                     seed=Seed,
                                     include_stop_str_in_output=True
)
FinalSamplingParameters = SamplingParams(temperature=1.0, 
                                     top_p=1.0, 
                                     max_tokens=1024,
                                     min_tokens=SamplingMinTokens,
                                     n=1,
                                     stop=["</answer>"],
                                     seed=Seed,
                                     include_stop_str_in_output=True
)
RewardFn = drgrpo_grader.r1_zero_reward_fn
os.environ["WANDB_API_KEY"] = "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"
wandb.login()
wandb.init(
    project="CS336-Assignment5-Alignment-EI",
    name=f"EI_Rollout{Rollout}BatchSize{BatchSize}",
    config={
        "learning_rate":LearningRate,
        "batchsize":BatchSize,
        "epochs":Epoch
    }
)
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")

model = AutoModelForCausalLM.from_pretrained(ModelPath,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2").to(TrainingDevice)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LearningRate,
    weight_decay=0.0,
    betas=(0.9, 0.95)
)
print("loading optimizer successful. Start loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ModelPath)
llm = utils.init_vllm(
model_id=ModelPath,
device=InferenceDevice,
seed=InferenceSeed,
gpu_memory_utilization=InferenceGpuMemoryUti
)

training_set = pd.read_parquet(path=DatasetTrain) # df with index of prompt and answer
test_set = pd.read_parquet(path=DatasetTest).sample(n=128)
test_prompts = [prompt for prompt in test_set["prompt"]]
test_answers = [answer for answer in test_set["answer"]]

train_step = 0
for epoch in range(Epoch):
    utils.load_policy_into_vllm_instance(policy=model, llm=llm)
    test_dict = sft.log_huge_generations(llm, FinalSamplingParameters, model, tokenizer, test_prompts, 
                             test_answers, RewardFn, MicroBatchSize
    )
    wandb.log({
    "eval_step":epoch,
    "eval/loss":test_dict["eval_loss"],
    "eval/acc":test_dict["eval_acc"],
    "eval/entropy": test_dict["eval_entropy"],
    "eval/avg_rsp_len":test_dict["eval_rsp_len"],
    "eval/avg_rsp_len_true":test_dict["eval_rsp_len_true"],
    "eval/avg_rsp_len_false":test_dict["eval_rsp_len_false"]
})
    for step in range(NEiSteps):
        training_samples = training_set.sample(BatchSize)
        sample_prompts = [question for question in training_samples["prompt"]]
        sample_answers = [question for question in training_samples["answer"]]
        llm_outputs = llm.generate(sample_prompts, SamplingParameters)
        training_prompts, training_outputs = [], []
        for j in range(BatchSize):
            ground_truth = sample_answers[j]
            for k in range(G):
                generated_text = llm_outputs[j].outputs[k].text
                reward_dict = RewardFn(generated_text, ground_truth)
                if reward_dict["reward"] == 1.0:
                    training_prompts.append(sample_prompts[j])
                    training_outputs.append(generated_text)
        tokenized_training = utils.tokenize_prompt_and_output(training_prompts, training_outputs, tokenizer)
        print(f"Epoch{epoch}-step{step} got {len(tokenized_training["input_ids"])} training samples")
        for j in range(0, len(tokenized_training["input_ids"])-MicroBatchSize+1, MicroBatchSize):
            input_ids = tokenized_training["input_ids"][j:j+MicroBatchSize].to(TrainingDevice)
            labels = tokenized_training["labels"][j:j+MicroBatchSize].to(TrainingDevice)
            masks = tokenized_training["response_mask"][j:j+MicroBatchSize].to(TrainingDevice)
            log_probs = sft.get_response_log_probs(model, input_ids, labels)["log_probs"]
            accumulation_steps = len(tokenized_training)//MicroBatchSize
            loss, metadata = sft.sft_microbatch_train_step(log_probs, masks, GradientAccumulationSteps)
            train_step += 1
            wandb.log({
                "train_step": train_step,
                "train/loss": loss * accumulation_steps
            })
            if j % GradientAccumulationSteps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                print("epoch", epoch, "loss", loss)
        if j % GradientAccumulationSteps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            print("epoch", epoch, "loss", loss) 

final_test_outputs = llm.generate(test_prompts, FinalSamplingParameters)
total_reward = 0.0
for idx, output in enumerate(final_test_outputs):
    text = output.outputs[0].text
    total_reward += RewardFn(text, test_answers[idx])["reward"]
last_acc = total_reward / len(test_prompts)
print("last test acc:", last_acc)
wandb.config.update({"last acc": last_acc})

final_path = os.path.join(OutPutDir, f"Batchsize{BatchSize}Rollout{Rollout}")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)