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



ModelPath = "/root/shared-nvme/Qwen2.5"
DatasetTrain = "/root/shared-nvme/gsm8k/main/train-prompt_and_output.parquet"
DatasetTest = "/root/shared-nvme/gsm8k/main/test-prompt_and_answer.parquet"
OutPutDir = "/root/shared-nvme/sft_checkpoints"
FinalTestPath = "/root/assignment5-alignment/Comparasion"
LearningRate = 1e-5
Epoch = 64
BatchSize = 2
GradientAccumulationSteps = 8
UniqueSamples = 7144
OurPutPath = os.path.join(OutPutDir, f"UniqueSample{UniqueSamples}.parquet")
TrainingDevice = torch.device("cuda:0")
InferenceDevice = torch.device("cuda:1")
InferenceSeed = 42
InferenceGpuMemoryUti = 0.75
InferSize = BatchSize * GradientAccumulationSteps
SamplingParameters = SamplingParams(temperature=1.0, 
                                     top_p=1.0, max_tokens=1024,
                                     stop=["</answer>"],
                                     include_stop_str_in_output=True)
RewardFn = drgrpo_grader.r1_zero_reward_fn
os.environ["WANDB_API_KEY"] = "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"
wandb.login()
wandb.init(
    project="CS336-Assignment5-Alignment",
    name=f"UniqueSample{UniqueSamples}",
    config={
        "UniqueSamples": UniqueSamples,
        "learning_rate":LearningRate,
        "batchsize":BatchSize,
        "epochs":Epoch
    }
)
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")


def training():
    print("Start loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(ModelPath,
                                torch_dtype=torch.bfloat16,
                                attn_implementation="flash_attention_2").to(TrainingDevice)
    # model.gradient_checkpointing_enable()
    print("Loading policy model successful! Start Loading optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LearningRate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )
    print("loading optimizer successful. Start loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ModelPath)
    dataset_train = utils.SFTDataset(DatasetTrain, unique_samples=UniqueSamples)
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=BatchSize,
                                  shuffle=True,
                                  collate_fn=partial(utils.collate_fn_sft, tokenizer=tokenizer))
    df_infer = pd.read_parquet(path=DatasetTest)
    infer_prompts_all = [prompt for prompt in df_infer["prompt"]]
    infer_outputs_all = [output for output in df_infer["answer"]]
    sample_indices = random.sample(range(len(infer_prompts_all)), InferSize)
    #we need to make infer size equal to batchsize cause the infer stage need to run policy model
    #so it can't be larger than batchsize, or it will out of memory
    llm = utils.init_vllm(
    model_id=ModelPath,
    device=InferenceDevice,
    seed=InferenceSeed,
    gpu_memory_utilization=InferenceGpuMemoryUti
    )
    infer_prompts = [infer_prompts_all[i] for i in sample_indices]
    infer_outputs = [infer_outputs_all[i] for i in sample_indices]
    train_step = 0
    print("We are ready, start to train")
    for epoch in range(Epoch):
        for idx, batch in enumerate(dataloader_train):
            input_ids, labels, masks = batch["input_ids"].to(TrainingDevice), \
                batch["labels"].to(TrainingDevice), batch["response_mask"].to(TrainingDevice)
            log_probs = sft.get_response_log_probs(model=model,
                                       input_ids=input_ids,
                                       labels=labels,
                                       return_token_entropy=False)["log_probs"]
            loss, metadata_dict = sft.sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=masks,
                gradient_accumulation_steps=GradientAccumulationSteps,
                normalize_constant=1.0)
            
            train_step += 1
            wandb.log({
                "train_step": train_step,
                "train/loss":loss * GradientAccumulationSteps
            })
            if (idx + 1) % GradientAccumulationSteps == 0:
                print(f"epoch {epoch} idx {idx} loss {loss}")
                optimizer.step()
                optimizer.zero_grad()
        if (idx + 1) % GradientAccumulationSteps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch {epoch} idx {idx} loss {loss}")
        if (epoch+1) % (Epoch//4) != 0:
            continue
        utils.load_policy_into_vllm_instance(
            policy=model,
            llm=llm
        )
        total_metrics = {
            "entropy": [],
            "loss": [],
            "avg_rsp_len": [],
            "avg_rsp_len_true": [],
            "avg_rsp_len_false": [],
        }
        for i in range(0, len(infer_prompts)-BatchSize, BatchSize):
            infer_prompt = infer_prompts[i : i + BatchSize]
            infer_output = infer_outputs[i : i + BatchSize]
            infer_dict = sft.log_generations(
                vllm_model=llm,
                sampling_params=SamplingParameters,
                policy_model=model,
                tokenizer=tokenizer,
                prompts=infer_prompt,
                ground_truths=infer_output,
                reward_fn=RewardFn
            )
            total_metrics["entropy"].append(infer_dict["entropy"])
            total_metrics["loss"].append(infer_dict["loss"])
            total_metrics["avg_rsp_len"].append(infer_dict["avg_rsp_len"])
            total_metrics["avg_rsp_len_true"].append(infer_dict["avg_rsp_len_true"])
            total_metrics["avg_rsp_len_false"].append(infer_dict["avg_rsp_len_false"])
        eval_loss = sum(total_metrics["loss"]) / len(total_metrics["loss"])
        eval_entropy = sum(total_metrics["entropy"]) / len(total_metrics["entropy"])
        eval_rsp_len = sum(total_metrics["avg_rsp_len"]) / len(total_metrics["avg_rsp_len"])
        eval_rsp_len_true = sum(total_metrics["avg_rsp_len_true"]) / len(total_metrics["avg_rsp_len_true"])
        eval_rsp_len_false = sum(total_metrics["avg_rsp_len_false"]) / len(total_metrics["avg_rsp_len_false"])

        wandb.log({
            "eval_step":epoch,
            "eval/loss":eval_loss,
            "eval/entropy": eval_entropy,
            "eval/avg_rsp_len":eval_rsp_len,
            "eval/avg_rsp_len_true":eval_rsp_len_true,
            "eval/avg_rsp_len_false":eval_rsp_len_false
        })
        print(f"epoch {epoch} eval loss {infer_dict["loss"]}")
        # checkpoint_path = os.path.join(OutPutDir, f"epoch_{epoch}")
        # os.makedirs(checkpoint_path, exist_ok=True)
        # model.save_pretrained(checkpoint_path)
        # tokenizer.save_pretrained(checkpoint_path)
        # print(f"Saved epoch {epoch} model to path {checkpoint_path}")
    # --- SAVE FINAL MODEL ---
    final_path = os.path.join(OutPutDir, f"Epoch{Epoch}Uniquesamples{UniqueSamples}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training complete. Final model saved.")
    test_df = baseline.evaluate_llm(
        vllm_model=llm,
        reward_fn=drgrpo_grader.r1_zero_reward_fn,
        datapath="/root/shared-nvme/gsm8k/main/test-preprocessed.parquet",
        eval_sampling_params=SamplingParameters,
        output_path=OurPutPath
    )
    stat_dict = utils.statistics(test_df)
    wandb.config.update({"both_right_rate":stat_dict["both_right_rate"]})

        
if __name__ == "__main__":
    training()
