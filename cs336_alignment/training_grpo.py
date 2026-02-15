import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment import utils, drgrpo_grader, grpo, sft
from vllm import SamplingParams
import pandas as pd
import os
import wandb




RolloutBatchSize = 256
GroupSize = 8
RolloutQuestions = RolloutBatchSize // GroupSize
MicroBatchSize = 2
GradientAccumulationSteps = RolloutBatchSize // MicroBatchSize

Steps = 500


TrainBatchSize = RolloutBatchSize
EpochsPerRolloutBatch = 1

AdvantageEps = 1e-6
lr= 1e-5

RewardFn = drgrpo_grader.r1_zero_reward_fn

LossType = "reinforce_with_baseline"

ModelPath = "/root/shared-nvme/sft_checkpoints/Epoch64Uniquesamples1024/"
DatasetTrain = "/root/shared-nvme/gsm8k/main/train-prompt_and_answer.parquet"
DatasetTest = "/root/shared-nvme/gsm8k/main/test-prompt_and_answer.parquet"
OutPutDir = "/root/shared-nvme/grpo_checkpoints"

TrainingDevice = torch.device('cuda:0')

SamplingParameters = SamplingParams(temperature=1.0, 
                                     top_p=1.0, 
                                     max_tokens=1024,
                                     min_tokens=4,
                                     n=GroupSize,
                                     stop=["</answer>"],
                                     seed=4,
                                     include_stop_str_in_output=True
)
ValidationgParameters = SamplingParams(temperature=1.0, 
                                     top_p=1.0, 
                                     max_tokens=1024,
                                     min_tokens=4,
                                     n=1,
                                     stop=["</answer>"],
                                     seed=4,
                                     include_stop_str_in_output=True
)

os.environ["WANDB_API_KEY"] = "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"
wandb.login()
wandb.init(
    project="CS336-Assignment5-Alignment",
    name="GRPO",
    config={
        "loss type": LossType,
        "learning_rate":lr,
        "Rollout_batchsize":RolloutBatchSize,
        "Total_steps":Steps
    }
)
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")


def main(Epoch: int = Steps, normalize_by_std: bool = True, loss_type: str = LossType):
    policy_model = AutoModelForCausalLM.from_pretrained(ModelPath,
                                                      torch_dtype=torch.bfloat16,
                                                      attn_implementation="flash_attention_2").to(TrainingDevice)
    
    policy_model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(ModelPath)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr, weight_decay=0.0, betas=(0.9,0.95))
    llm = utils.init_vllm(ModelPath, device=torch.device('cuda:1'), seed=42, gpu_memory_utilization=0.75)
    
    training_df = pd.read_parquet(path=DatasetTrain)
    validation_df = pd.read_parquet(path=DatasetTest).sample(n=64)
    validation_prompts = [prompt for prompt in validation_df["prompt"]]
    validation_ground_truths = [answer for answer in validation_df["answer"]]
    for epoch in range(Epoch):
        print(f"Starting epoch {epoch}...")
        training_df_rollout = training_df.sample(n=RolloutQuestions)
        sample_questions = [question for question in training_df_rollout["prompt"]]
        sample_answers = [answer for answer in training_df_rollout["answer"]]
        utils.load_policy_into_vllm_instance(policy=policy_model, llm=llm)
        llm_outputs = llm.generate(sample_questions, SamplingParameters)
        training_prompts = [question for question in sample_questions for _ in range(GroupSize)]
        training_answers = [answer for answer in sample_answers for _ in range(GroupSize)]
        training_outputs = [llm_outputs[i].outputs[j].text for i in range(RolloutQuestions) for j in range(GroupSize)]
        advantages, raw_rewards, _ = grpo.compute_group_normalized_rewards(RewardFn, training_outputs, training_answers,
                                                                        GroupSize, AdvantageEps, normalize_by_std)
        advantages, raw_rewards = advantages.to(TrainingDevice), raw_rewards.to(TrainingDevice)
        train_acc = raw_rewards.mean().item()
        wandb.log({
            "train_step":epoch,
            "train/acc":train_acc
        })
        print(f"training_set acc {train_acc}")
        tokenized_training = utils.tokenize_prompt_and_output(training_prompts, training_outputs, tokenizer)
        for k in range(EpochsPerRolloutBatch):
            total_loss = 0.0
            for i in range(0, RolloutBatchSize, MicroBatchSize):
                input_ids = tokenized_training["input_ids"][i:i+MicroBatchSize].to(TrainingDevice)
                labels = tokenized_training["labels"][i:i+MicroBatchSize].to(TrainingDevice)
                masks = tokenized_training["response_mask"][i:i+MicroBatchSize].to(TrainingDevice)
                log_probs = sft.get_response_log_probs(policy_model, input_ids, labels)["log_probs"]
                loss, metadata = grpo.grpo_microbatch_train_step(log_probs, masks, GradientAccumulationSteps,
                                                                 loss_type, raw_rewards[i:i+MicroBatchSize].unsqueeze_(-1),
                                                                   advantages[i:i+MicroBatchSize].unsqueeze_(-1))
                total_loss += loss.item()
            print(f"total training loss {total_loss}")
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        # validation
        utils.load_policy_into_vllm_instance(policy=policy_model, llm=llm)
        if (epoch+1) % 10 == 0:
            llm_val_outputs = llm.generate(validation_prompts, ValidationgParameters)
            llm_val_text = [output.outputs[0].text for output in llm_val_outputs]
            rewards = [RewardFn(text, answer)["reward"] for text, answer in zip(llm_val_text, validation_ground_truths)]
            avg_reward = sum(rewards) / len(rewards)
            print(f"avg_validation_reward {avg_reward}")
            wandb.log({
                "eval_step": epoch,
                "eval/reward": avg_reward
            })
    final_path = os.path.join(OutPutDir, f"total_steps{Steps}{LossType}")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

        
main()
