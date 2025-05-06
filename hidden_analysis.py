import json
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

# 从 原始数据文件/评估结果文件 中提取出 正确/错误 的回答样本
def generate_math_data(data_dir, data_path):
    correct, incorrect = [], []
    with open(data_path) as f: # 原始数据文件train.jsonl
        data = f.readlines()
        data = [json.loads(line) for line in data]
    with open(f"{data_dir}/math_eval.jsonl") as f: # 评估结果文件math_eval.jsonl
        eval = f.readlines()
        eval = [json.loads(line) for line in eval]
    # 将 data 截取至与 eval 相同长度，确保两者对齐
    data = data[:len(eval)]
    # d 是原始问题数据，e 是评估数据
    for d, e in zip(data, eval):
        local_correct, local_incorrect = [], [] # 分别存正确和错误的回答
        prompt = e["prompt"]
        assert d["problem"] == e["problem"]
        # 挨个看模型的每次回答和评分
        for o, c in zip(e["model_generation"], e["all_eval"]):
            if c: # 如果正确，就加入正确列表，否则加入错误列表
                local_correct.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
            else:
                local_incorrect.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
        correct.extend(local_correct)
        incorrect.extend(local_incorrect)
    return correct, incorrect
    



# 分析文本中的推理步骤，识别 反思/转换 行为
def generate_index(text, tokenizer, split_id, think_only=True):
    # text输入文本 split_id分隔符的令牌ID
    check_words=["verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect", "Let me check", "seems right"]
    check_prefix = ["Wait"]
    swicth_words = ["think differenly", "another way", "another approach", "another method", "another solution", "another strategy", "another technique"]
    switch_prefix = ["Alternatively"]
    
    tokens = tokenizer.encode(text)
    # 是否只处理 cot 之间的内容
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return [], [], []
    
        start = tokens.index(think_begin_id)+1
        if think_end_id not in tokens[start:]:
            end=len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0
    # 找到 cot 里换行符的位置(思维过程的边界)+最后一个
    # [(0, 100), (1, 20), (2, 200), (3, 300), (4, 20), (5, 400)]---> [1,4,6] 20作为\n
    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = [] # 存储每一步的起始位置 [1, 3, 5, 7]每步的起始位置
    check_index = [] # 存储 反思 的步骤索引 [1, 3]第2步和第4步是反思
    switch_index = [] # 存储 转换 的步骤索引 [2]第3步是转换
    # 遍历由 index 划分的段落
    for i in range(len(index)-1):
        step_index.append(index[i]+start)
        step = think_tokens[index[i]+1:index[i+1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        if any([step.lower().startswith(p.lower()) for p in check_prefix]) or any([w.lower() in step.lower() for w in check_words]):
                check_index.append(i)
        elif any([step.lower().startswith(p.lower()) for p in switch_prefix]) or any([w.lower() in step.lower() for w in swicth_words]):
            switch_index.append(i)
    return step_index, check_index, switch_index

# 加载模型，对每个样本进行推理，提取各层隐藏状态并保存
def generate(model_path, data, save_dir):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left" # 适合因果语言模型
    # 分词器没有填充标记(pad_token)，用结束标记(eos_token)替代
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # 获取分词器的词汇表 vocab，提取包含 ĊĊ（表示\n\n）的令牌ID，存入 split_id
    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"]+d["response"] for d in data]

    layer_num = model.config.num_hidden_layers+1 # 获取模型隐藏层数量(包含输入层)
    hidden_dict=[{} for _ in range(layer_num)] # 存储每层的隐藏状态数据
    # 遍历每个提示，生成对应的隐藏状态
    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        # 将提示分词，生成PyTorch张量
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        # {'input_ids': tensor([[151646, 151644, 16141, ...]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad(): # 不计算梯度
            output = model(**tokenized_batch, output_hidden_states=True)
            hidden_states = output.hidden_states
            hidden_states = [h.detach().cpu() for h in hidden_states] # 将隐藏状态从GPU移动到CPU,节省显存
        layer_num = len(hidden_states)
        step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=think_only)
        # 将索引转tensor
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)
        for i in range(layer_num):
            h = hidden_states[i][0] # 取出第 i 层的隐藏状态张量的第一个序列
            step_h = h[step_index]
            hidden_dict[i][k] = {"step":step_h, "check_index": check_index, "switch_index": switch_index}
        del hidden_states
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt") # 将包含隐藏状态的hidden_dict保存为.pt
    json.dump(prompts, open(f"{save_dir}/prompts.json", "w"))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()
    correct, incorrect = generate_math_data(data_dir=args.data_dir, data_path=args.data_path)
    if args.type == "correct":
        data = correct
    else:
        data = incorrect
    save_dir = f"{args.data_dir}/hidden_{args.type}"
    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start+args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    print(save_dir)
    generate(args.model_path, data, save_dir)

# generate_math_data() → 提取正确/错误回答
#       ↓
# generate() → 加载模型和数据
#       ↓
# 逐条处理 prompt + response
#       ↓
# 调用 generate_index() 解析推理步骤边界
#       ↓
# 获取各层隐藏状态
#       ↓
# 将每一步的隐藏状态和反思索引保存进 hidden_dict
#       ↓
# 保存为 hidden.pt 和 prompts.json

# hidden_dict = [
#     layer_0: {             # 输入嵌入层
#         sample_0: {        # k是样本编号，v是该样本在该层的推理信息
#             "step": tensor([5, 4096]),     # 5 个推理步骤，每个 4096 维
#             "check_index": [1, 3],         # 第 1 和第 3 步是反思
#             "switch_index": [2]            # 第 2 步是换种思路
#         },
#         sample_1: {...},
#         ...
#     },
#     layer_1: {              # 第一层 Transformer
#         sample_0: {
#             "step": tensor([5, 4096]),
#             "check_index": [1, 3],
#             "switch_index": [2]
#         },
#         ...
#     },
#     ...
# ]