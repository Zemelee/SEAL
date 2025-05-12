import torch
import os
import argparse


# 从隐藏状态数据中生成推理引导向量

def load_data(data_dir, prefixs, layer_num=29, max_examples=None):
    data_paths = [os.path.join(data_dir, f"hidden_{p}", "hidden.pt") for p in prefixs]
    switch = [[] for _ in range(layer_num)] # 列表，元素对应模型的层，存储该层中某类推理块的隐藏状态
    check = [[] for _ in range(layer_num)]
    other = [[] for _ in range(layer_num)]
    # prefixs = ["correct", "incorrect"]
    for i, data_path in enumerate(data_paths):
        data = torch.load(data_path, weights_only=False)  # 数组: 每个元素dict，代表某一层的所有样本
        # 遍历每一层、每个样本，提取：所有 step 的隐藏状态 反思/转换 的位置索引
        for l in range(layer_num):
            layer_data = data[l]  # 某一层的所有样本 {sample_id: {}...}
            for k in layer_data:  # 样本编号k
                if max_examples is not None and max_examples > 0 and k >= max_examples:
                    continue
                h = layer_data[k]["step"]  # torch.Size([28, 1536]) # 28个step
                check_index = layer_data[k]["check_index"]  # l层的样本k中哪些 step 是 反思类型 tensor([19, 24]) shape(2)
                switch_index = layer_data[k]["switch_index"]  # tensor([25]) shape(1)
                check[l].append(h[check_index])
                switch[l].append(h[switch_index])
                # 筛选出“执行类思维“
                all_indices = torch.arange(h.shape[0]) # 所有推理块的索引列表
                mask = ~(
                    torch.isin(all_indices, check_index)
                    | torch.isin(all_indices, switch_index)
                ) # ~取反, 取出不在check_index和switch_index中的索引
                other[l].append(h[mask])
    # 找到每层每个样本的思维块的位置后 
    for l in range(layer_num):
        # 对每一层的数据进行拼接，dim=0，按第0维(行)拼接
        check[l] = torch.cat(check[l], dim=0)
        switch[l] = torch.cat(switch[l], dim=0)
        other[l] = torch.cat(other[l], dim=0)
    # 将所有层的数据堆叠起来
    check = torch.stack(check, dim=0)
    switch = torch.stack(switch, dim=0)
    other = torch.stack(other, dim=0)
    return check, switch, other


# 根据 load_data 生成向量并保存为pt
def generate_vector_switch_check(
    data_dir, prefixs, layers, save_prefix, overwrite=False
):
    if isinstance(layers, int):
        layers = [layers]
    max_layer = max(layers)
    # 分别包含所有层的反思/转换/执行的隐藏状态
    check, switch, other = load_data(
        data_dir=data_dir, prefixs=prefixs, layer_num=max_layer + 1
    ) #shape =  [Layer_Num, Num_Thoughts_In_Layer, Hidden_Dim]
    save_dir = os.path.join(data_dir, f"vector_{save_prefix}")
    os.makedirs(save_dir, exist_ok=True)
    #  从各层的隐藏状态中提取一个 steering_vector
    for layer in layers: # 只针对第20层
        layer_check = check[layer]
        layer_switch = switch[layer]
        layer_other = other[layer]
        # 反思/转换块的平均表示 - 执行类推理块的平均表示
        steer_vec = torch.cat([layer_check, layer_switch], dim=0).mean(
            dim=0
        ) - layer_other.mean(dim=0) # “冗余推理”与“有效推理”的方向差异
        save_path = os.path.join(
            save_dir, f"layer_{layer}_transition_reflection_steervec.pt"
        )
        if not os.path.exists(save_path) or overwrite:
            torch.save(steer_vec, save_path)
        else:
            print(f"{save_path} already exists")
        print(f"layer {layer} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--prefixs", type=str, nargs="+", default=["correct_0_500", "incorrect_0_500"]
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[20])
    parser.add_argument("--save_prefix", type=str, default="500_500")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_vector_switch_check(
        data_dir=args.data_dir,
        prefixs=args.prefixs,
        layers=args.layers,
        save_prefix=args.save_prefix,
        overwrite=args.overwrite,
    )
    # generate_vector_switch_check(
    #     data_dir="results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000",
    #     prefixs=["correct_0_500", "incorrect_0_500"],
    #     layers=20,
    #     save_prefix="500_500",
    # )
