import math
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

'''注意力函数'''
def attention(q,k,v,dropout=None):
    d_k=q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    p_atten=scores.softmax(dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, v), p_atten

'''多头注意力机制'''
#临时配置文件，包含输入序列长度，输入词向量维度，多头注意力头数，dropout比例
parser = argparse.ArgumentParser(description='多头注意力参数配置')
parser.add_argument('--dim',type=int,default=512)
parser.add_argument('--n_embd',type=int,default=512)
parser.add_argument('--n_heads',type=int,default=8)
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--max_seq_len',type=int,default=128)
parser.add_argument('--n_layer',type=int,default=6)
# 补充MLP所需的hidden_dim参数（通常是n_embd的4倍）
parser.add_argument('--hidden_dim', type=int, default=2048)
args = parser.parse_args()#解析参数方法
class MultiHeadAttention(nn.Module):
    def __init__(self,args,is_causal=False):
        super().__init__()
        #首先计算分给每一个头的维度，并且给head和head_dim赋值属性
        self.head_dim = args.dim//args.n_heads
        self.n_heads = args.n_heads

        #因为有8个头，每个头都需要一个qwv矩阵，所以根据初始输入的512维词向量需要经过linear转换为64维然后循环8次
        #但是循环8次计算量太大了，所以把原来的8个64维的拼接在一起
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)

        #拼接完再内积，相当于在一个大矩阵里面进行多个头运算不用拆开了
        self.wo = nn.Linear(self.head_dim*self.n_heads,args.dim,bias=False)

        #定义dropout
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)

        #掩码注意力
        self.is_causal = is_causal
        if is_causal:
            mask = torch.full((1,1,args.max_seq_len,args.max_seq_len),fill_value=float('-inf'))
            mask = torch.triu(mask,diagonal=1)#把矩阵转换成“上三角矩阵”（只保留对角线及以下，对角线以上设为负无穷）
            # 注册为模型的缓冲区,不会被优化器更新参数
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz,seqlen,_ = q.shape
        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        #一个把注意力头变成一个整体
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        #计算注意力分数
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        #将多头注意力拼接起来
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1,2).contiguous().view(bsz,seqlen,-1)

        #最后进行线性投影
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

'''前馈神经网络'''
#全连接网络→relu全连接网络→dropout
class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super().__init__()
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)

    def forward(self,x):
        return self.dropout(self.w2(F.relu(self.w1(x))))


'''LayerNorm'''
class LayerNorm(nn.Module):
    def __init__(self,features_dim,eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features_dim))
        self.b_2 = nn.Parameter(torch.zeros(features_dim))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1,keepdim=True) # mean: [bsz, max_len, 1]
        # 注意这⾥也在最后⼀个维度发⽣了⼴播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


'''Encoder'''
#Encoder由n个EncoderLayer组成
#每一个EncoderLayer先经过一个层归一化，然后后经过注意力
class EncoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        # ⼀个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(dim=args.n_embd,hidden_dim=args.hidden_dim,dropout=args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # ⾃注意⼒
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经⽹络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
#开始搭建Encoder
class Encoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        #普通列表不能被注必须经过册
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        #分别通过 N 层 Encoder Layer，最后
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

'''Dencoder'''
#一个layer有三个layernorm，分别在Mask Attention 、Attention 、 MLP之前
class DecoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args,is_causal=True)
        self.attention = MultiHeadAttention(args)
        self.feed_forward = MLP(dim=args.n_embd,hidden_dim=args.hidden_dim,dropout=args.dropout)
    def forward(self,x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码⾃注意⼒
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意⼒
        # 2. 交叉注意力计算（对接Encoder输出，获取输入序列的语义信息）
        # 【Q/K/V标注】：交叉注意力中，Q来自Decoder，K/V来自Encoder输出enc_out
        # - Q (Query)：norm_x → Decoder已生成序列的归一化语义（要查询信息的主体）
        # - K (Key)：enc_out → Encoder编码后的输入序列语义（用于匹配Q的参考信息）
        # - V (Value)：enc_out → Encoder编码后的输入序列语义（要提取给Q的核心信息）
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经⽹络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
#搭建decoder
class Decoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        #"Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

# -------------------------- 优化后的测试代码 --------------------------
if __name__ == "__main__":
    # 1. 实例化模型组件
    encoder_model = Encoder(args)
    decoder_model = Decoder(args)


    # 2. 参数量统计函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    enc_params = count_parameters(encoder_model)
    dec_params = count_parameters(decoder_model)
    total_params = enc_params + dec_params

    # 3. 打印统计报告
    print("=" * 80)
    print(f"{'Transformer 架构参数统计':^70}")
    print("=" * 80)
    table_format = "{:<25} | {:<20} | {:<15}"
    print(table_format.format("组件名称", "参数数量 (Num)", "参数占比"))
    print("-" * 80)
    print(table_format.format(f"Encoder ({args.n_layer}层)", f"{enc_params:,}",
                              f"{(enc_params / total_params) * 100:.2f}%"))
    print(table_format.format(f"Decoder ({args.n_layer}层)", f"{dec_params:,}",
                              f"{(dec_params / total_params) * 100:.2f}%"))
    print("-" * 80)
    print(table_format.format("总计 (Total)", f"{total_params:,}", "100.00%"))

    # 基础物理内存换算 (仅限 Float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
    print(f"预计权重文件大小 (Float32): {memory_mb:.2f} MB")

    # 4. 前向传播形状验证
    print("\n" + "=" * 80)
    print(f"{'组件张量形状校验 (Forward Check)':^70}")
    print("=" * 80)

    # 构造标准测试数据
    batch_size = 2
    seq_len = args.max_seq_len
    dim = args.n_embd

    # 测试 Encoder
    x_enc = torch.randn(batch_size, seq_len, dim)
    enc_out = encoder_model(x_enc)
    print(f"Encoder: 输入 {list(x_enc.shape)} -> 输出 {list(enc_out.shape)} ✅")

    # 测试 Decoder
    x_dec = torch.randn(batch_size, seq_len, dim)
    dec_out = decoder_model(x_dec, enc_out)
    print(f"Decoder: 输入 {list(x_dec.shape)} -> 输出 {list(dec_out.shape)} ✅")

    # 5. 梯度传导测试
    loss = dec_out.mean()
    loss.backward()

    encoder_grad = any(p.grad is not None for p in encoder_model.parameters())
    decoder_grad = any(p.grad is not None for p in decoder_model.parameters())

    print("-" * 80)
    print(f"梯度检查: Decoder [{'OK' if decoder_grad else 'FAIL'}] | Encoder [{'OK' if encoder_grad else 'FAIL'}]")
    print("=" * 80)