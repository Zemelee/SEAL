# 加载数据集（MATH500、GSM等）
# 使用指定的模型生成答案
# 对生成的答案进行后处理（比如提取最终答案、修剪多余部分）
# 保存预测结果并评估
import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial


import os
import gc
# 禁止分词器的并行化
from get_math_results import main as eval_main
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"
exact_match = evaluate.load("exact_match")


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    """
    对给定的 logits 进行调整。
    Args:
        token_ids (torch.Tensor): 输入的 token IDs。
        logits (torch.Tensor): 输入的 logits。
        adjust_ids (torch.Tensor): 需要调整的 token IDs。
        values (float): 调整的值。
        max_len (int, optional): 最大长度。默认为 -1。
    Returns:
        torch.Tensor: 调整后的 logits。

    """
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits


# 保留前缀，即去掉开头的指令、问题和注释
def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

# 从模型生成的字符串中提取最终答案
def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

# 提取最后一个数字
def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    # 加载数据集
    if args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":extract_box(example["solution"]),
            })
    elif args.dataset == "MATH_train":
        data_path = "data/MATH/train.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":extract_box(example["solution"]), # 标准答案
                })
    elif args.dataset in ["GSM", "GSM_train"]:
        if args.dataset == "GSM_train":
            data_path = "data/gsm/train.jsonl"
        else:
            data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError("Dataset not supported")
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载分词器tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # 构建提示（Prompt）
    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        prompt =  prefix+"Question: " + example["question"].strip()+"\nAnswer: "
        if args.use_chat_format: # 对话格式处理
            # 如果模型是gemma或者deepseek，则使用chat_template
            if "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    # 加载模型
    model = LLM(
        model=args.model_name_or_path,
        tokenizer=(
            args.tokenizer_name_or_path
            if args.tokenizer_name_or_path
            else args.model_name_or_path
        ),
        swap_space=8,
        gpu_memory_utilization=0.8,
        enable_lora=args.peft is not None,
        tensor_parallel_size=torch.cuda.device_count(),
        max_lora_rank=64,
        max_model_len=args.max_tokens + 500,
    )
    # 设置生成参数
    if not args.logit_adjustment:
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    # 生成答案                                    
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
    # 处理输出
    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)

    outputs = [[trim_output(o) for o in output] for output in result]

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in zip(test_data, outputs, prompts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )


    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)

    main(args) # 加载数据集和模型，生成答案，并保存为predictions.jsonl
    # 读取predictions, 给出评估结果
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
