// T5Encoder.swift — the T5-base (v1.0) encoder SAM-Audio conditions on: `transformers.T5EncoderModel("t5-base")`'s
// `last_hidden_state`. Compact keys written by Tools/convert (shared, relative_attention_bias, blocks.N.*, final_norm).
//
// T5 specifics: layer norm is scale-only with no mean subtraction; attention has no 1/√d scaling; one relative-position bias
// (32 buckets, bidirectional, max distance 128) is shared by every block; the feed-forward is wo(relu(wi x)).
import Foundation
import MLX
import MLXNN

final class T5Norm: Module {
    let weight: MLXArray
    init(_ dim: Int) { weight = MLXArray.ones([dim]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + Float(1e-6)) * weight }
}

final class T5Block: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: T5Norm
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "ff_norm") var ffNorm: T5Norm
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    let heads: Int
    let dKV: Int

    init(dim: Int, heads: Int, dKV: Int, dFF: Int) {
        self.heads = heads
        self.dKV = dKV
        self._attnNorm.wrappedValue = T5Norm(dim)
        self._q.wrappedValue = Linear(dim, heads * dKV, bias: false)
        self._k.wrappedValue = Linear(dim, heads * dKV, bias: false)
        self._v.wrappedValue = Linear(dim, heads * dKV, bias: false)
        self._o.wrappedValue = Linear(heads * dKV, dim, bias: false)
        self._ffNorm.wrappedValue = T5Norm(dim)
        self._wi.wrappedValue = Linear(dim, dFF, bias: false)
        self._wo.wrappedValue = Linear(dFF, dim, bias: false)
    }

    /// `bias`: (1, heads, S, S) position bias plus the additive padding mask.
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, s, heads, dKV]).transposed(0, 2, 1, 3) }
        let n = attnNorm(x)
        let scores = matmul(split(q(n)), split(k(n)).transposed(0, 1, 3, 2)) + bias
        let attn = matmul(softmax(scores, axis: -1), split(v(n))).transposed(0, 2, 1, 3).reshaped([b, s, heads * dKV])
        let h = x + o(attn)
        return h + wo(relu(wi(ffNorm(h))))
    }
}

public final class T5Encoder: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "relative_attention_bias") var relativeBias: Embedding
    @ModuleInfo(key: "blocks") var blocks: [T5Block]
    @ModuleInfo(key: "final_norm") var finalNorm: T5Norm
    let buckets: Int
    let maxDistance: Int

    public init(_ c: SAMAudioConfig.TextEncoder) {
        buckets = c.relative_buckets
        maxDistance = c.relative_max_distance
        self._shared.wrappedValue = Embedding(embeddingCount: c.vocab, dimensions: c.dim)
        self._relativeBias.wrappedValue = Embedding(embeddingCount: c.relative_buckets, dimensions: c.heads)
        self._blocks.wrappedValue = (0..<c.layers).map { _ in T5Block(dim: c.dim, heads: c.heads, dKV: c.d_kv, dFF: c.d_ff) }
        self._finalNorm.wrappedValue = T5Norm(c.dim)
    }

    public static func load(bundle: URL, config: SAMAudioConfig, dtype: DType? = nil) throws -> T5Encoder {
        let enc = T5Encoder(config.text_encoder)
        try WeightIO.apply(WeightIO.load(bundle.appendingPathComponent("text_encoder.safetensors"), dtype: dtype), to: enc, component: "text_encoder")
        return enc
    }

    /// transformers' `_relative_position_bucket`, bidirectional. Integer math on the host: S is a prompt, a handful of tokens.
    func bucketTable(_ s: Int) -> MLXArray {
        let nb = buckets / 2, maxExact = nb / 2
        var out = [Int32](repeating: 0, count: s * s)
        for i in 0..<s {
            for j in 0..<s {
                let rel = j - i
                var bucket = rel > 0 ? nb : 0
                let n = abs(rel)
                if n < maxExact {
                    bucket += n
                } else {
                    let large = maxExact + Int(Foundation.log(Float(n) / Float(maxExact)) / Foundation.log(Float(maxDistance) / Float(maxExact)) * Float(nb - maxExact))
                    bucket += min(large, nb - 1)
                }
                out[i * s + j] = Int32(bucket)
            }
        }
        return MLXArray(out, [s, s])
    }

    /// `ids`: (B, S) int32; `mask`: (B, S) bool, true = token. Returns (B, S, dim).
    public func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        let s = ids.dim(1)
        var h = shared(ids)
        let position = relativeBias(bucketTable(s)).transposed(2, 0, 1).expandedDimensions(axis: 0)  // (1, heads, S, S)
        let pad = MLX.where(mask, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude)).expandedDimensions(axes: [1, 2])
        let bias = (position + pad).asType(h.dtype)
        for block in blocks { h = block(h, bias: bias) }
        return finalNorm(h)
    }
}
