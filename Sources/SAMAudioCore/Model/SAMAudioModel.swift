// SAMAudioModel.swift — the velocity field: SAM-Audio's `forward` (sam_audio/model/model.py @ bb4c699) for the text + span path.
//
// `align_masked_video` is NOT optional on this path. Upstream calls it on an all-zero video feature when there is no video,
// which adds `tanh(gate) · LayerNorm(conv.bias)` to every frame (gate −0.167 in sam-audio-base). mlx-audio's port drops the
// module and is 19–25 dB from upstream per forward for it (WIP/sam-audio-port/MEASUREMENTS.md, Neywa-Labs/mlx-audio#1176).
// The bundle keeps gate, conv.bias and the layer norm; the conv weight is only for real video features.
import Foundation
import MLX
import MLXNN

final class AlignBias: Module {
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(_ dim: Int) { self._bias.wrappedValue = MLXArray.zeros([dim]) }
}

/// The text-path remainder of upstream's AlignModalities.
final class AlignMaskedVideo: Module {
    @ParameterInfo(key: "gate") var gate: MLXArray
    @ModuleInfo(key: "conv") var conv: AlignBias
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    init(_ dim: Int) {
        self._gate.wrappedValue = MLXArray.zeros([1])
        self._conv.wrappedValue = AlignBias(dim)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
    }
    /// What upstream adds to every frame when the video feature is all zeros: (dim,).
    func constant() -> MLXArray { tanh(gate) * layerNorm(conv.bias) }
}

final class EmbedAnchors: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ParameterInfo(key: "gate") var gate: MLXArray
    @ModuleInfo(key: "proj") var proj: Linear
    init(numAnchors: Int, embeddingDim: Int, outDim: Int) {
        self._embed.wrappedValue = Embedding(embeddingCount: numAnchors + 1, dimensions: embeddingDim)
        self._gate.wrappedValue = MLXArray.zeros([1])
        self._proj.wrappedValue = Linear(embeddingDim, outDim, bias: false)
    }
    /// `anchorIDs`: (B, A) token ids; `alignment`: (B, T) index into each row of ids.
    func callAsFunction(_ x: MLXArray, anchorIDs: MLXArray, alignment: MLXArray) -> MLXArray {
        let ids = takeAlong(anchorIDs, alignment, axis: 1)
        return x + tanh(gate) * proj(embed(ids))
    }
}

public struct Conditioning {
    /// (B, T, 2C): the codec features, duplicated for target and residual.
    public var audioFeatures: MLXArray
    /// (B, S, 768) T5 features and the (B, S) attention mask (true = token).
    public var textFeatures: MLXArray
    public var textMask: MLXArray
    /// (B, A) anchor token ids (0 null, 1 "+", 2 "−", 3 pad) and the (B, T) frame → anchor index map.
    public var anchorIDs: MLXArray
    public var anchorAlignment: MLXArray
    /// (B, T), true = a real frame; nil when nothing is padded.
    public var audioPadMask: MLXArray?

    public init(audioFeatures: MLXArray, textFeatures: MLXArray, textMask: MLXArray, anchorIDs: MLXArray, anchorAlignment: MLXArray, audioPadMask: MLXArray? = nil) {
        self.audioFeatures = audioFeatures
        self.textFeatures = textFeatures
        self.textMask = textMask
        self.anchorIDs = anchorIDs
        self.anchorAlignment = anchorAlignment
        self.audioPadMask = audioPadMask
    }
}

public final class SAMAudioModel: Module {
    @ModuleInfo(key: "transformer") public var transformer: DiT
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "align_masked_video") var alignMaskedVideo: AlignMaskedVideo
    @ModuleInfo(key: "embed_anchors") var embedAnchors: EmbedAnchors
    @ModuleInfo(key: "memory_proj") var memoryProj: Linear
    let dim: Int

    public init(_ c: SAMAudioConfig) {
        dim = c.transformer.dim
        self._transformer.wrappedValue = DiT(c.transformer)
        self._proj.wrappedValue = Linear(c.in_channels, c.transformer.dim, bias: true)
        self._alignMaskedVideo.wrappedValue = AlignMaskedVideo(c.transformer.dim)
        self._embedAnchors.wrappedValue = EmbedAnchors(numAnchors: c.num_anchors, embeddingDim: c.anchor_embedding_dim, outDim: c.transformer.dim)
        self._memoryProj.wrappedValue = Linear(c.text_encoder.dim, c.transformer.dim, bias: true)
    }

    public static func load(bundle: URL, config: SAMAudioConfig, dtype: DType? = nil) throws -> SAMAudioModel {
        let model = SAMAudioModel(config)
        try WeightIO.apply(WeightIO.load(bundle.appendingPathComponent("model.safetensors"), dtype: dtype), to: model, component: "model")
        return model
    }

    /// cos ‖ sin of `t · exp(−ln(10000) · i / (dim / 2))`: upstream's model-level SinusoidalEmbedding with `pos = time`.
    func timeMemory(_ time: MLXArray) -> MLXArray {
        let half = dim / 2
        let inv = exp(MLXArray((0..<half).map { -Foundation.log(10000.0) * Float($0) / Float(half) }))
        let e = time.asType(.float32).expandedDimensions(axis: 1) * inv.expandedDimensions(axis: 0)
        return concatenated([cos(e), sin(e)], axis: -1).expandedDimensions(axis: 1)  // (B, 1, dim)
    }

    static func additive(_ mask: MLXArray?) -> MLXArray? {
        guard let mask else { return nil }
        let neg = MLXArray(-Float.infinity)
        return MLX.where(mask, MLXArray(Float(0)), neg).expandedDimensions(axes: [1, 2])  // (B, 1, 1, keys)
    }

    /// One evaluation of the velocity field. `noisy`: (B, T, 2C); `time`: (B,).
    public func callAsFunction(noisy: MLXArray, time: MLXArray, _ c: Conditioning) -> MLXArray {
        let x = concatenated([noisy, MLXArray.zeros(like: c.audioFeatures), c.audioFeatures], axis: 2)
        var aligned = proj(x) + alignMaskedVideo.constant().asType(x.dtype)
        aligned = embedAnchors(aligned, anchorIDs: c.anchorIDs, alignment: c.anchorAlignment)
        let memory = memoryProj(c.textFeatures) + timeMemory(time).asType(x.dtype)
        return transformer(aligned, time: time, memory: memory, selfMask: Self.additive(c.audioPadMask), memoryMask: Self.additive(c.textMask))
    }
}
