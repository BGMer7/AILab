import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        
        self.h = heads
        self.d_k = d_model // heads
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 拼接多头注意力输出
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)

        return output
    
    
def test_multihead_attention():
    """测试多头注意力模块"""
    print("🚀 开始测试多头注意力模块")
    print("="*50)
    
    # 设置参数
    batch_size = 2
    seq_length = 5
    d_model = 512
    heads = 8
    
    print(f"📊 测试参数:")
    print(f"   - batch_size: {batch_size}")
    print(f"   - seq_length: {seq_length}")
    print(f"   - d_model: {d_model}")
    print(f"   - heads: {heads}")
    print(f"   - d_k (每个头的维度): {d_model // heads}")
    
    # 创建模型
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    print(f"\n✅ 创建多头注意力模块成功")
    
    # 创建测试数据
    # 模拟句子: "我 爱 学习 机器 学习"
    input_data = torch.randn(batch_size, seq_length, d_model)
    print(f"\n📥 输入数据形状: {input_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        # 自注意力：Q=K=V
        output = mha(input_data, input_data, input_data)
        print(f"📤 输出数据形状: {output.shape}")
        
        # 验证形状是否正确
        assert output.shape == input_data.shape, "输出形状应该与输入相同"
        print("✅ 形状验证通过")
    
    # 打印参数数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n📈 模型参数:")
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - Q线性层: {d_model * d_model:,}")
    print(f"   - K线性层: {d_model * d_model:,}")
    print(f"   - V线性层: {d_model * d_model:,}")
    print(f"   - 输出层: {d_model * d_model:,}")
    
    return mha, input_data, output

def visualize_attention_effect():
    """可视化注意力效果"""
    print("\n🎯 测试注意力效果")
    print("="*50)
    
    # 使用小一点的参数便于观察
    batch_size = 1
    seq_length = 4
    d_model = 64
    heads = 4
    
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    
    # 创建有意义的测试数据
    # 模拟4个词的句子，每个词有不同的特征模式
    test_input = torch.zeros(batch_size, seq_length, d_model)
    
    # 第1个词：前16维为1
    test_input[0, 0, :16] = 1.0
    # 第2个词：中间16维为1  
    test_input[0, 1, 16:32] = 1.0
    # 第3个词：后16维为1
    test_input[0, 2, 32:48] = 1.0
    # 第4个词：混合特征
    test_input[0, 3, [0, 16, 32, 48]] = 1.0
    
    print("📊 输入特征模式:")
    for i in range(seq_length):
        non_zero = torch.nonzero(test_input[0, i]).flatten()
        print(f"   词{i+1}: 非零位置 {non_zero[:5].tolist()}{'...' if len(non_zero) > 5 else ''}")
    
    # 前向传播
    with torch.no_grad():
        output = mha(test_input, test_input, test_input)
        
        print(f"\n📤 输出统计:")
        print(f"   - 输入范围: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"   - 输出范围: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   - 输出均值: {output.mean():.3f}")
        print(f"   - 输出标准差: {output.std():.3f}")
        
        # 计算每个位置的输出变化
        print(f"\n🔄 注意力效果分析:")
        for i in range(seq_length):
            input_norm = torch.norm(test_input[0, i])
            output_norm = torch.norm(output[0, i])
            print(f"   词{i+1}: 输入范数={input_norm:.3f} -> 输出范数={output_norm:.3f}")

def test_with_mask():
    """测试掩码功能"""
    print("\n🎭 测试掩码功能")
    print("="*50)
    
    batch_size = 1
    seq_length = 5
    d_model = 1024
    heads = 4
    
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    input_data = torch.randn(batch_size, seq_length, d_model)
    
    # 创建掩码：假设最后两个位置是padding
    mask = torch.ones(batch_size, seq_length)
    mask[0, 3:] = 0  # 掩盖最后两个位置
    
    print(f"📋 掩码模式: {mask[0].tolist()}")
    print("   (1表示有效位置，0表示padding位置)")
    
    with torch.no_grad():
        # 不使用掩码
        output_no_mask = mha(input_data, input_data, input_data)
        
        # 使用掩码
        output_with_mask = mha(input_data, input_data, input_data, mask=mask)
        
        print(f"\n📊 掩码效果对比:")
        for i in range(seq_length):
            no_mask_norm = torch.norm(output_no_mask[0, i])
            with_mask_norm = torch.norm(output_with_mask[0, i])
            mask_status = "有效" if mask[0, i] == 1 else "掩盖"
            print(f"   位置{i+1}({mask_status}): 无掩码={no_mask_norm:.3f}, 有掩码={with_mask_norm:.3f}")

def compare_different_heads():
    """比较不同头数的效果"""
    print("\n🔢 比较不同头数的效果")
    print("="*50)
    
    d_model = 256
    seq_length = 6
    batch_size = 1
    
    head_configs = [1, 2, 4, 8]
    input_data = torch.randn(batch_size, seq_length, d_model)
    
    results = {}
    for heads in head_configs:
        mha = MultiHeadAttention(heads=heads, d_model=d_model)
        with torch.no_grad():
            output = mha(input_data, input_data, input_data)
            results[heads] = {
                'output_std': output.std().item(),
                'output_mean': output.mean().item(),
                'params': sum(p.numel() for p in mha.parameters())
            }
    
    print(f"📊 不同头数对比:")
    print(f"{'头数':<6} {'输出标准差':<12} {'输出均值':<12} {'参数量':<12}")
    print("-" * 50)
    for heads in head_configs:
        r = results[heads]
        print(f"{heads:<6} {r['output_std']:<12.4f} {r['output_mean']:<12.4f} {r['params']:<12,}")

if __name__ == "__main__":
    # 运行所有测试
    print("🎉 多头注意力模块综合测试")
    print("="*70)
    
    # 基础功能测试
    mha, input_data, output = test_multihead_attention()
    
    # 注意力效果可视化
    visualize_attention_effect()
    
    # 掩码功能测试
    test_with_mask()
    
    # 不同头数对比
    compare_different_heads()
    
    print("\n🎊 测试完成！多头注意力模块运行正常")
    print("="*70)

 