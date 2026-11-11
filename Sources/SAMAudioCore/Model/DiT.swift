// DiT.swift — SAM-Audio's diffusion transformer, translated 1:1 from facebookresearch/sam-audio @ bb4c699
// (sam_audio/model/transformer.py, patcher.py, rope.py). Module keys are the upstream checkpoint's, so the bundle needs no
// rename. Everything is channels-last (B, T, C).
//
// Three upstream details that are shape-safe to get wrong:
//  * `Attention.reshape_heads` splits (B, T, C) as (B, T, C / heads, heads) and moves the LAST axis to the front — the head
//    index is the fast axis of the projection, not the slow one — while the output is merged heads-major.
//  * q/k RMS norms act per head (dim = head_dim), before RoPE; cross-attention has the norms and no RoPE.
//  * the final modulation uses the timestep embedding itself; the blocks use `t_block(silu(t))`.
import Foundation
import MLX
import MLXNN

/// `x * rsqrt(mean(x², -1) + eps) * weight`, with the norm taken in fp32 as upstream does (`self._norm(x.float())`): in
/// fp16 the squares overflow and the half-precision tier separates nothing (V2 S2b, first fp16 run).
final class RMSNorm: Module {
    let weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// `w2(silu(w1 x) * w3 x)` — upstream's ProjectionLayer and FeedForward in their SwiGLU form (no biases).
final class SwiGLU: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear
    init(inDim: Int, hidden: Int, outDim: Int) {
        self._w1.wrappedValue = Linear(inDim, hidden, bias: false)
        self._w2.wrappedValue = Linear(hidden, outDim, bias: false)
        self._w3.wrappedValue = Linear(inDim, hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

/// Rotary table for (B, H, L, E): pairs (x0, x1) -> (x0·cos − x1·sin, x0·sin + x1·cos). Not a Module, so the table is no key.
final class RotaryTable {
    let cosT: MLXArray  // (maxLen, E / 2)
    let sinT: MLXArray
    init(headDim: Int, maxLen: Int, theta: Float) {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
        let freqs = 1.0 / pow(MLXArray(theta), idx / Float(headDim))
        let t = MLXArray((0..<maxLen).map { Float($0) })
        let ang = outer(t, freqs)
        cosT = cos(ang)
        sinT = sin(ang)
        eval(cosT, sinT)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, l, e) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let p = x.reshaped([b, h, l, e / 2, 2])
        let x0 = p[.ellipsis, 0], x1 = p[.ellipsis, 1]
        let c = cosT[0..<l].asType(x.dtype), s = sinT[0..<l].asType(x.dtype)
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped([b, h, l, e])
    }
}

final class Attention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, eps: Float) {
        self.heads = heads
        self.headDim = dim / heads
        self._wq.wrappedValue = Linear(dim, dim, bias: false)
        self._wk.wrappedValue = Linear(dim, dim, bias: false)
        self._wv.wrappedValue = Linear(dim, dim, bias: false)
        self._wo.wrappedValue = Linear(dim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dim / heads, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dim / heads, eps: eps)
    }

    /// (B, T, C) -> (B, heads, T, C / heads), with the head index on the projection's fast axis (upstream `reshape_heads`).
    private func splitHeads(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), x.dim(1), headDim, heads]).transposed(0, 3, 1, 2)
    }

    /// `additiveMask`: broadcastable to (B, heads, T, S); 0 where a key may be attended, −inf where it is padding.
    func callAsFunction(_ x: MLXArray, cross: MLXArray? = nil, additiveMask: MLXArray? = nil, rope: RotaryTable? = nil) -> MLXArray {
        let src = cross ?? x
        var q = qNorm(splitHeads(wq(x)))
        var k = kNorm(splitHeads(wk(src)))
        let v = splitHeads(wv(src))
        if let rope {
            q = rope(q)
            k = rope(k)
        }
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * (1.0 / Float(headDim).squareRoot())
        if let additiveMask { scores = scores + additiveMask.asType(scores.dtype) }
        let out = matmul(softmax(scores, axis: -1), v)  // (B, H, T, hd)
        return wo(out.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), heads * headDim]))
    }
}

final class DiTBlock: Module {
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "cross_attention") var crossAttention: Attention
    @ModuleInfo(key: "feed_forward") var feedForward: SwiGLU
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    let dim: Int

    init(dim: Int, heads: Int, ffnHidden: Int, eps: Float) {
        self.dim = dim
        self._attention.wrappedValue = Attention(dim: dim, heads: heads, eps: eps)
        self._crossAttention.wrappedValue = Attention(dim: dim, heads: heads, eps: eps)
        self._feedForward.wrappedValue = SwiGLU(inDim: dim, hidden: ffnHidden, outDim: dim)
        self._attentionNorm.wrappedValue = RMSNorm(dim, eps: eps)
        self._ffnNorm.wrappedValue = RMSNorm(dim, eps: eps)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([6, dim])
    }

    /// `t6`: (B, 6·dim) from `t_block`. Order: shift/scale/gate for self-attention, then for the feed-forward.
    func callAsFunction(_ x: MLXArray, memory: MLXArray, t6: MLXArray, selfMask: MLXArray?, memoryMask: MLXArray?, rope: RotaryTable) -> MLXArray {
        let b = scaleShiftTable.expandedDimensions(axis: 0) + t6.reshaped([x.dim(0), 6, dim])
        func row(_ i: Int) -> MLXArray { b[0..., i..<(i + 1), 0...] }  // (B, 1, dim)
        let hAttn = attention(attentionNorm(x) * (1 + row(1)) + row(0), additiveMask: selfMask, rope: rope)
        var h = x + hAttn * row(2)
        h = h + crossAttention(h, cross: memory, additiveMask: memoryMask)
        let hFF = feedForward(ffnNorm(h) * (1 + row(4)) + row(3))
        return h + hFF * row(5)
    }
}

/// GroupNorm(1) → SiLU → Conv1d(k = 3, same padding). Upstream ConvBlock1d.
final class ConvBlock1d: Module {
    @ModuleInfo(key: "groupnorm") var groupnorm: GroupNorm
    @ModuleInfo(key: "project") var project: Conv1d
    init(_ channels: Int) {
        self._groupnorm.wrappedValue = GroupNorm(groupCount: 1, dimensions: channels, eps: 1e-5, affine: true, pytorchCompatible: true)
        self._project.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { project(silu(groupnorm(x.asType(.float32)).asType(x.dtype))) }
}

final class ResnetBlock1d: Module {
    @ModuleInfo(key: "block1") var block1: ConvBlock1d
    @ModuleInfo(key: "block2") var block2: ConvBlock1d
    init(_ channels: Int) {
        self._block1.wrappedValue = ConvBlock1d(channels)
        self._block2.wrappedValue = ConvBlock1d(channels)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(block1(x)) + x }
}

/// Upstream Patcher at patch_size 1: one residual block.
final class Patcher: Module {
    @ModuleInfo(key: "block") var block: ResnetBlock1d
    init(_ channels: Int) { self._block.wrappedValue = ResnetBlock1d(channels) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x) }
}

final class Projection: Module {
    @ModuleInfo(key: "projection") var projection: SwiGLU
    init(inDim: Int, outDim: Int) { self._projection.wrappedValue = SwiGLU(inDim: inDim, hidden: outDim, outDim: outDim) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

public final class DiT: Module {
    @ModuleInfo(key: "layers") var layers: [DiTBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "output") var output: Linear
    @ModuleInfo(key: "x_embedder") var xEmbedder: Patcher
    @ModuleInfo(key: "y_embedder") var yEmbedder: Projection
    @ModuleInfo(key: "t_embedder") var tEmbedder: Projection
    @ModuleInfo(key: "t_block") var tBlock: Linear
    @ParameterInfo(key: "final_layer_scale_shift_table") var finalTable: MLXArray
    let rope: RotaryTable
    let freqDim: Int
    let dim: Int

    public init(_ c: SAMAudioConfig.Transformer) {
        dim = c.dim
        freqDim = c.frequency_embedding_dim
        var hidden = Int(2 * (c.ffn_exp * c.dim) / 3) * c.ffn_dim_multiplier
        hidden = c.multiple_of * ((hidden + c.multiple_of - 1) / c.multiple_of)
        self._layers.wrappedValue = (0..<c.n_layers).map { _ in DiTBlock(dim: c.dim, heads: c.n_heads, ffnHidden: hidden, eps: c.norm_eps) }
        self._norm.wrappedValue = RMSNorm(c.dim, eps: c.norm_eps)
        self._output.wrappedValue = Linear(c.dim, c.out_channels, bias: false)
        self._xEmbedder.wrappedValue = Patcher(c.dim)
        self._yEmbedder.wrappedValue = Projection(inDim: c.dim, outDim: c.dim)
        self._tEmbedder.wrappedValue = Projection(inDim: c.frequency_embedding_dim, outDim: c.dim)
        self._tBlock.wrappedValue = Linear(c.dim, 6 * c.dim, bias: true)
        self._finalTable.wrappedValue = MLXArray.zeros([2, c.dim])
        rope = RotaryTable(headDim: c.dim / c.n_heads, maxLen: c.max_positions, theta: Float(max(10000, 2 * c.max_positions)))
    }

    /// cos ‖ sin of `t · exp(−ln(10000) · i / half)`.
    func timestepFeatures(_ t: MLXArray) -> MLXArray {
        let half = freqDim / 2
        let freqs = exp(MLXArray((0..<half).map { -Foundation.log(10000.0) * Float($0) / Float(half) }))
        let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return concatenated([cos(args), sin(args)], axis: -1)
    }

    /// `x`: (B, T, dim) aligned inputs; `memory`: (B, S, dim); masks are additive, broadcastable to (B, 1, 1, keys), or nil.
    public func callAsFunction(_ x: MLXArray, time: MLXArray, memory: MLXArray, selfMask: MLXArray? = nil, memoryMask: MLXArray? = nil) -> MLXArray {
        var h = xEmbedder(x)
        let t = tEmbedder(timestepFeatures(time).asType(x.dtype))  // (B, dim)
        let t6 = tBlock(silu(t))
        let y = yEmbedder(memory)
        for layer in layers {
            h = layer(h, memory: y, t6: t6, selfMask: selfMask, memoryMask: memoryMask, rope: rope)
        }
        let table = finalTable.expandedDimensions(axis: 0) + t.expandedDimensions(axis: 1)  // (B, 2, dim)
        h = norm(h) * (1 + table[0..., 1..<2, 0...]) + table[0..., 0..<1, 0...]
        return output(h)
    }
}
