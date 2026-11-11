// Pipeline.swift — a mixture and a description in, the described sound (target) and everything else (residual) out.
// Upstream `SAMAudio.separate` for the text + span path: encode → 16-step midpoint flow from noise → decode both halves.
//
// Prompts are lowercase noun or verb phrases ("speech", "dog barking"): the model was trained on that form and misses on
// sentences (upstream README; WIP/sam-audio-eval V0c). Outputs are 48 kHz mono and carry the codec's watermark.
import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// A time span where the described sound is (`present`) or is not: the model's "+" / "−" anchors.
public struct SeparationSpan: Sendable, Equatable {
    public let start: Double
    public let end: Double
    public let present: Bool
    public init(start: Double, end: Double, present: Bool = true) {
        self.start = start
        self.end = end
        self.present = present
    }
}

public struct SeparationOutput {
    /// (samples,) each, 48 kHz mono, the mixture's length.
    public let target: MLXArray
    public let residual: MLXArray
}

public final class SAMAudioPipeline {
    public let config: SAMAudioConfig
    public let model: SAMAudioModel
    public let textEncoder: T5Encoder
    public let codec: DACVAE
    let tokenizer: any Tokenizer

    /// `modelDType`: the DiT's dtype (nil = as stored). The T5 encoder and the codec always run fp32 (T5 overflows in half
    /// precision, and it is 0.44 GB).
    public init(bundle: URL, modelDType: DType? = nil) async throws {
        config = try SAMAudioConfig.load(bundle.appendingPathComponent("config.json"))
        model = try SAMAudioModel.load(bundle: bundle, config: config, dtype: modelDType)
        textEncoder = try T5Encoder.load(bundle: bundle, config: config, dtype: .float32)
        codec = try DACVAE.load(bundle: bundle, dtype: .float32)
        tokenizer = try await AutoTokenizer.from(modelFolder: bundle)
    }

    /// T5 token ids for one prompt, with the closing `</s>`, truncated to the encoder's limit.
    public func tokens(_ prompt: String) -> [Int32] {
        var ids = tokenizer.encode(text: prompt)
        if ids.last != 1 { ids.append(1) }
        if ids.count > config.text_encoder.max_length { ids = Array(ids.prefix(config.text_encoder.max_length - 1)) + [1] }
        return ids.map(Int32.init)
    }

    /// Upstream `Batch.process_anchors` for one item: ids [null, pad, …one per span], and each frame's index into them.
    static func anchors(_ spans: [SeparationSpan], frames: Int) -> (ids: MLXArray, alignment: MLXArray) {
        var ids: [Int32] = [0, 3]
        var alignment = [Int32](repeating: 0, count: frames)
        for s in spans {
            let a = min(frames, max(0, DACVAE.frames(forSamples: Int((s.start * Double(DACVAE.sampleRate)).rounded(.up)))))
            let b = min(frames, max(0, DACVAE.frames(forSamples: Int((s.end * Double(DACVAE.sampleRate)).rounded(.up)))))
            if b > a { for i in a..<b { alignment[i] = Int32(ids.count) } }
            ids.append(s.present ? 1 : 2)
        }
        return (MLXArray(ids, [1, ids.count]), MLXArray(alignment, [1, frames]))
    }

    public func conditioning(wav: MLXArray, prompt: String, spans: [SeparationSpan] = []) -> Conditioning {
        let ids = tokens(prompt)
        let idArray = MLXArray(ids, [1, ids.count])
        let mask = MLXArray.ones([1, ids.count], dtype: .bool)
        let text = textEncoder(idArray, mask: mask)
        let feats = codec.encode(wav.reshaped([1, -1]))
        let a = Self.anchors(spans, frames: feats.dim(1))
        let dtype = model.transformer.parameters().flattened().first?.1.dtype ?? .float32
        let c = Conditioning(audioFeatures: concatenated([feats, feats], axis: 2).asType(dtype), textFeatures: text.asType(dtype), textMask: mask,
                             anchorIDs: a.ids, anchorAlignment: a.alignment, audioPadMask: nil)
        eval(c.audioFeatures, c.textFeatures)
        return c
    }

    /// `wav`: (samples,) 48 kHz mono. `noise`: (1, frames, 256) to pin the draw; otherwise drawn from `seed`.
    public func separate(wav: MLXArray, prompt: String, spans: [SeparationSpan] = [], seed: UInt64 = 0, steps: Int = 16,
                         method: FlowSolver.Method = .midpoint, noise: MLXArray? = nil, message: MLXArray? = nil,
                         onStep: ((Int) throws -> Void)? = nil, onDecode: ((Int, Int) throws -> Void)? = nil) throws -> SeparationOutput {
        guard wav.ndim == 1, wav.dim(0) > 0 else { throw SAMAudioError.invalidInput("wav must be (samples,) and non-empty") }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw SAMAudioError.invalidInput("empty description") }
        let c = conditioning(wav: wav, prompt: prompt, spans: spans)
        let start = noise ?? MLXRandom.normal(c.audioFeatures.shape, key: MLXRandom.key(seed))
        let latent = try FlowSolver.solve(model, noise: start.asType(c.audioFeatures.dtype), c, steps: steps, method: method, onStep: onStep)
        let half = DACVAE.latentDim
        // Decode the two halves one after the other: the decoder's 48 kHz activations are the pipeline's memory peak.
        // The watermark's 16-bit message is drawn from the seed too (its own key), so a seeded separation is reproducible to
        // the sample. Upstream draws it from the global generator.
        let msg = (message ?? MLXRandom.randInt(low: 0, high: 2, [1, DACVAE.messageBits], key: MLXRandom.key(seed &+ 0x5A4D_A0D1))).reshaped([1, -1])
        var halves = [MLXArray]()
        for (i, part) in [latent[0..., 0..., 0..<half], latent[0..., 0..., half...]].enumerated() {
            try onDecode?(i + 1, 2)
            let w = codec.decode(part.asType(.float32), message: msg)
            eval(w)
            halves.append(w)
            MLX.Memory.clearCache()
        }
        let wavs = concatenated(halves, axis: 0)
        let n = min(wav.dim(0), wavs.dim(1))
        let out = SeparationOutput(target: wavs[0, 0..<n], residual: wavs[1, 0..<n])
        eval(out.target, out.residual)
        return out
    }
}
