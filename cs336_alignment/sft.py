import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: [batchsize, seq_length, vocab_size]
    logits_stable = logits - torch.max(logits, dim=-1, keepdim=True).values
    probability = torch.exp(logits_stable) / torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True)
    return torch.sum(-probability * torch.log(probability), dim=-1, keepdim=False)

def get_response_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor,\
                           return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze_(-1)).squeeze_(-1)
    results = {"log_probs": log_probs}
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)
    return results

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float=1.0,
        dim: int | None = None,
) -> torch.Tensor:
    needed_sum_tensor = torch.where(mask==1, tensor, 0)
    return torch.sum(needed_sum_tensor, dim=dim) / normalize_constant
    
def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant) / gradient_accumulation_steps / policy_log_probs.shape[0]
    loss.backward()
    return loss, {"loss": loss.detach()}


def log_generations(vllm_model, sampling_params, policy_model, tokenizer, prompts, ground_truths, reward_fn):
    llm_outputs = vllm_model.generate(prompts, sampling_params)
    # in sampling_params, we set: logprobs=n, will return top-n logprob
    generated_text = [output.outputs[0].text for output in llm_outputs]
    tokenized = tokenize_prompt_and_output(prompts, generated_text, tokenizer)
    input_ids = tokenized["input_ids"].to(policy_model.device)
    labels = tokenized["labels"].to(policy_model.device)
    response_mask = tokenized["response_mask"].to(policy_model.device)
    with torch.no_grad():
        # Get log probs and entropy for the FULL sequence
        log_probs_dict = get_response_log_probs(
            model=policy_model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True
        )
        
        # Extract full sequence entropy: shape (batch, seq_len)
        full_entropy = log_probs_dict["token_entropy"]
        
        # 4. Filter for Response Only
        # We use masked_mean (Problem 7.2) to average only the response tokens
        # response_mask is 1 for response, 0 for prompt/padding
        avg_response_entropy = masked_normalize(
            tensor=full_entropy, 
            mask=response_mask, 
            normalize_constant=1.0,
            dim=None # Average over all response tokens in the batch
        ) / torch.sum(response_mask)
        loss = -masked_normalize(log_probs_dict["log_probs"], response_mask) / log_probs_dict["log_probs"].shape[0]


    generated_tokens = [output.outputs[0].token_ids for output in llm_outputs]
    token_length = [len(t) for t in generated_tokens]
    rewards = [reward_fn(text, answer) for text, answer in zip(generated_text, ground_truths)]
    format_reward = [reward["format_reward"] for reward in rewards]
    answer_reward = [reward["answer_reward"] for reward in rewards]
    reward = [reward["reward"] for reward in rewards]

    df = pd.DataFrame({"prompts":prompts, "ground_truths":ground_truths, "generated_text":generated_text,\
                       "format_reward":format_reward, "answer_reward":answer_reward, "reward":reward,\
                        "length":token_length})
# 5. Safe Statistics
    # FIX 3: Use boolean masking instead of groupby to prevent crashes
    avg_rsp_len = np.mean(token_length) if token_length else 0.0
    
    # Filter using the DataFrame
    true_rows = df[df["reward"] == 1.0]
    acc = len(true_rows) / len(df)
    false_rows = df[df["reward"] == 0.0]
    
    avg_rsp_len_true = true_rows["length"].mean() if not true_rows.empty else 0.0
    avg_rsp_len_false = false_rows["length"].mean() if not false_rows.empty else 0.0
    return {
        "metadata": df,
        "acc": acc,
        "loss": loss,
        "avg_rsp_len": avg_rsp_len,
        "avg_rsp_len_true": avg_rsp_len_true,
        "avg_rsp_len_false": avg_rsp_len_false,
        "entropy": avg_response_entropy
    }

def log_huge_generations(vllm_model, sampling_params, policy_model, tokenizer, prompts, ground_truths, reward_fn, micro_batchsize):
    total_metrics = {
    "entropy": [],
    "acc":[],
    "loss": [],
    "avg_rsp_len": [],
    "avg_rsp_len_true": [],
    "avg_rsp_len_false": []
    }
    for i in range(0, len(prompts)-micro_batchsize, micro_batchsize):
        infer_prompt = prompts[i : i + micro_batchsize]
        infer_output = ground_truths[i : i + micro_batchsize]
        infer_dict = log_generations(
            vllm_model=vllm_model,
            sampling_params=sampling_params,
            policy_model=policy_model,
            tokenizer=tokenizer,
            prompts=infer_prompt,
            ground_truths=infer_output,
            reward_fn=reward_fn
        )
        total_metrics["entropy"].append(infer_dict["entropy"])
        total_metrics["loss"].append(infer_dict["loss"])
        total_metrics["acc"].append(infer_dict["acc"])
        total_metrics["avg_rsp_len"].append(infer_dict["avg_rsp_len"])
        total_metrics["avg_rsp_len_true"].append(infer_dict["avg_rsp_len_true"])
        total_metrics["avg_rsp_len_false"].append(infer_dict["avg_rsp_len_false"])
    eval_loss = sum(total_metrics["loss"]) / len(total_metrics["loss"])
    eval_acc = sum(total_metrics["acc"]) / len(total_metrics["acc"])
    eval_entropy = sum(total_metrics["entropy"]) / len(total_metrics["entropy"])
    eval_rsp_len = sum(total_metrics["avg_rsp_len"]) / len(total_metrics["avg_rsp_len"])
    eval_rsp_len_true = sum(total_metrics["avg_rsp_len_true"]) / len(total_metrics["avg_rsp_len_true"])
    eval_rsp_len_false = sum(total_metrics["avg_rsp_len_false"]) / len(total_metrics["avg_rsp_len_false"])
    return {
            "eval_loss":eval_loss,
            "eval_acc":eval_acc,
            "eval_entropy": eval_entropy,
            "eval_rsp_len":eval_rsp_len,
            "eval_rsp_len_true":eval_rsp_len_true,
            "eval_rsp_len_false":eval_rsp_len_false
    }



if __name__ == "__main__":
    print("Hello, sft.py")