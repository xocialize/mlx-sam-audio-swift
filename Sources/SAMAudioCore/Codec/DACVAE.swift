// DACVAE.swift — SAM-Audio's audio codec: Meta's DAC-VAE (facebookresearch/dacvae @ 414c207) with its AudioSeal watermark
// branch, 48 kHz mono, hop 1920 (25 latent frames per second), 128-dim VAE mean as the latent.
//
// Translated from dacvae/model/dacvae.py and nn/layers.py. Weights come from Tools/convert/convert_codec.py under upstream's
// own key paths (weight_norm folded, convs in MLX layout); the layers are built from a `WeightBag` that must be consumed
// exactly — a key left over or missing throws. Everything is channels-last (B, T, C).
//
// The decoder ALWAYS watermarks (operator decision AB-D-0222, matching upstream's `Decoder.forward`): the waveform is
// `plain + 0.25 · mark(plain, message)`, with a random 16-bit message when the caller gives none. `decodePlain` exists for
// the parity lane only and is internal.
//
// Upstream's names are misleading in the watermark path: a decoder block's `upsample_group()` holds the STRIDED conv (it
// shortens the sequence) and `downsample_group()` holds the transposed conv (it lengthens it).
import Foundation
import MLX
import MLXNN
import MLXRandom

/// The converted tensors, handed out once each.
final class WeightBag {
    private var arrays: [String: MLXArray]
    private(set) var missing: [String] = []
    init(_ arrays: [String: MLXArray]) { self.arrays = arrays }
    func take(_ key: String) -> MLXArray {
        guard let v = arrays.removeValue(forKey: key) else {
            missing.append(key)
            return MLXArray.zeros([1])
        }
        return v
    }
    var unused: [String] { arrays.keys.sorted() }
}

private func snake(_ x: MLXArray, _ alpha: MLXArray) -> MLXArray {
    let s = sin(alpha * x)
    return x + (1 / (alpha + 1e-9)) * s * s
}

/// NormConv1d. `causalAuto` is upstream's `pad_mode="auto", causal=True`; otherwise `pad_mode="none"` (symmetric).
private struct Conv {
    let w: MLXArray, b: MLXArray
    let stride: Int, dilation: Int, causalAuto: Bool
    init(_ bag: WeightBag, _ key: String, stride: Int = 1, dilation: Int = 1, causalAuto: Bool = false) {
        w = bag.take(key + ".weight")
        b = bag.take(key + ".bias")
        self.stride = stride
        self.dilation = dilation
        self.causalAuto = causalAuto
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let k = w.dim(1)
        if !causalAuto {
            return conv1d(x, w, stride: stride, padding: (k - stride) * dilation / 2, dilation: dilation) + b
        }
        let length = x.dim(1)
        let effective = (k - 1) * dilation + 1
        let total = effective - stride
        let frames = Double(length - effective + total) / Double(stride) + 1
        let ideal = (Int(frames.rounded(.up)) - 1) * stride + (k - total)
        let extra = ideal - length
        let wide = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((total, max(extra, 0))), IntOrPair((0, 0))])
        return conv1d(wide, w, stride: stride, padding: 0, dilation: dilation) + b
    }
}

/// NormConvTranspose1d, kernel 2·stride.
private struct ConvT {
    let w: MLXArray, b: MLXArray
    let stride: Int, causalAuto: Bool
    init(_ bag: WeightBag, _ key: String, stride: Int, causalAuto: Bool = false) {
        w = bag.take(key + ".weight")
        b = bag.take(key + ".bias")
        self.stride = stride
        self.causalAuto = causalAuto
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if !causalAuto {
            return convTransposed1d(x, w, stride: stride, padding: (stride + 1) / 2, outputPadding: stride % 2) + b
        }
        let y = convTransposed1d(x, w, stride: stride, padding: 0) + b
        let k = w.dim(1)
        return y[0..., 0..<(y.dim(1) - (k - stride)), 0...]
    }
}

/// ResidualUnit in its Snake form (weight-normed, kernel 7, symmetric) or its ELU form (kernel 3, causal, half-width hidden).
private struct ResidualUnit {
    let alpha1: MLXArray?, alpha2: MLXArray?
    let c1: Conv, c2: Conv
    init(_ bag: WeightBag, _ key: String, dilation: Int) {
        alpha1 = bag.take(key + ".block.0.alpha")
        c1 = Conv(bag, key + ".block.1", dilation: dilation)
        alpha2 = bag.take(key + ".block.2.alpha")
        c2 = Conv(bag, key + ".block.3")
    }
    init(elu bag: WeightBag, _ key: String) {
        alpha1 = nil
        alpha2 = nil
        c1 = Conv(bag, key + ".block.1", causalAuto: true)
        c2 = Conv(bag, key + ".block.3", causalAuto: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let alpha1, let alpha2 { return x + c2(snake(c1(snake(x, alpha1)), alpha2)) }
        return x + c2(elu(c1(elu(x))))
    }
}

/// torch.nn.LSTM, unidirectional, batch-second in upstream; here (B, T, C). Gate order i, f, g, o; with the skip.
private struct SkipLSTM {
    let wih: [MLXArray], whh: [MLXArray], bias: [MLXArray]
    init(_ bag: WeightBag, _ key: String, layers: Int = 2) {
        wih = (0..<layers).map { bag.take("\(key).lstm.weight_ih_l\($0)") }
        whh = (0..<layers).map { bag.take("\(key).lstm.weight_hh_l\($0)") }
        bias = (0..<layers).map { bag.take("\(key).lstm.bias_l\($0)") }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        let (b, t) = (x.dim(0), x.dim(1))
        for l in wih.indices {
            let hidden = whh[l].dim(1)
            let pre = matmul(input, wih[l].T) + bias[l]  // (B, T, 4H)
            let whhT = whh[l].T
            var h = MLXArray.zeros([b, hidden], dtype: x.dtype), c = MLXArray.zeros([b, hidden], dtype: x.dtype)
            var outs = [MLXArray]()
            outs.reserveCapacity(t)
            for s in 0..<t {
                let g = pre[0..., s, 0...] + matmul(h, whhT)
                let i = sigmoid(g[0..., 0..<hidden]), f = sigmoid(g[0..., hidden..<(2 * hidden)])
                let n = tanh(g[0..., (2 * hidden)..<(3 * hidden)]), o = sigmoid(g[0..., (3 * hidden)...])
                c = f * c + i * n
                h = o * tanh(c)
                outs.append(h)
                if s % 128 == 127 { eval(h, c) }
            }
            input = stacked(outs, axis: 1)
            eval(input)
        }
        return input + x
    }
}

public final class DACVAE {
    public static let sampleRate = 48_000
    public static let hop = 1920
    public static let latentDim = 128
    public static let messageBits = 16
    static let rates = [2, 8, 10, 12]
    static let decoderRates = [12, 10, 8, 2]
    static let watermarkRates = [8, 5, 4, 2]
    static let alpha: Float = 0.25  // wm_channels / d_wm_out = 32 / 128

    // encoder
    private let encIn: Conv
    private let encBlocks: [(units: [ResidualUnit], alpha: MLXArray, down: Conv)]
    private let encAlpha: MLXArray
    private let encOut: Conv
    private let inProj: Conv
    // decoder, main path
    private let outProj: Conv
    private let decIn: Conv
    private let decBlocks: [(alpha: MLXArray, up: ConvT, units: [ResidualUnit])]
    private let outAlpha: MLXArray
    private let outConv: Conv
    // watermark
    private let markIn: Conv
    private let markShrink: [(unit: ResidualUnit, conv: Conv)]  // per decoder block: `upsample_group` (it shortens)
    private let markGrow: [(up: ConvT, unit: ResidualUnit)]     // per decoder block: `downsample_group` (it lengthens)
    private let markEncLSTM: SkipLSTM
    private let markEncOut: Conv
    private let messageTable: MLXArray
    private let markDecIn: Conv
    private let markDecLSTM: SkipLSTM
    private let markOut: Conv

    public init(arrays: [String: MLXArray]) throws {
        let bag = WeightBag(arrays)
        encIn = Conv(bag, "encoder.block.0")
        encBlocks = Self.rates.enumerated().map { i, s in
            let p = "encoder.block.\(i + 1).block"
            return (units: [ResidualUnit(bag, "\(p).0", dilation: 1), ResidualUnit(bag, "\(p).1", dilation: 3), ResidualUnit(bag, "\(p).2", dilation: 9)],
                    alpha: bag.take("\(p).3.alpha"), down: Conv(bag, "\(p).4", stride: s))
        }
        encAlpha = bag.take("encoder.block.5.alpha")
        encOut = Conv(bag, "encoder.block.6")
        inProj = Conv(bag, "quantizer.in_proj")
        outProj = Conv(bag, "quantizer.out_proj")
        decIn = Conv(bag, "decoder.model.0")
        var main = [(alpha: MLXArray, up: ConvT, units: [ResidualUnit])]()
        var shrink = [(unit: ResidualUnit, conv: Conv)](), grow = [(up: ConvT, unit: ResidualUnit)]()
        for (i, s) in Self.decoderRates.enumerated() {
            let p = "decoder.model.\(i + 1).block", ws = Self.watermarkRates[i]
            main.append((alpha: bag.take("\(p).0.alpha"), up: ConvT(bag, "\(p).1", stride: s),
                         units: [ResidualUnit(bag, "\(p).4", dilation: 1), ResidualUnit(bag, "\(p).5", dilation: 3), ResidualUnit(bag, "\(p).8", dilation: 9)]))
            grow.append((up: ConvT(bag, "\(p).3", stride: ws, causalAuto: true), unit: ResidualUnit(elu: bag, "\(p).6")))
            shrink.append((unit: ResidualUnit(elu: bag, "\(p).7"), conv: Conv(bag, "\(p).11", stride: ws, causalAuto: true)))
        }
        decBlocks = main
        markGrow = grow
        markShrink = shrink
        let wm = "decoder.wm_model"
        outAlpha = bag.take("\(wm).encoder_block.pre.0.alpha")
        outConv = Conv(bag, "\(wm).encoder_block.pre.1")
        markIn = Conv(bag, "\(wm).encoder_block.pre.3", causalAuto: true)
        markEncLSTM = SkipLSTM(bag, "\(wm).encoder_block.post.0")
        markEncOut = Conv(bag, "\(wm).encoder_block.post.2", causalAuto: true)
        messageTable = bag.take("\(wm).msg_processor.msg_processor.weight")
        markDecIn = Conv(bag, "\(wm).decoder_block.pre.0", causalAuto: true)
        markDecLSTM = SkipLSTM(bag, "\(wm).decoder_block.pre.1")
        markOut = Conv(bag, "\(wm).decoder_block.post.1", causalAuto: true)
        guard bag.missing.isEmpty && bag.unused.isEmpty else {
            throw SAMAudioError.keyContract(component: "codec", missing: bag.missing, unused: bag.unused)
        }
    }

    public static func load(bundle: URL, dtype: DType? = nil) throws -> DACVAE {
        try DACVAE(arrays: WeightIO.load(bundle.appendingPathComponent("codec.safetensors"), dtype: dtype))
    }

    /// Frames for a waveform of `samples` (upstream `wav_idx_to_feature_idx`).
    public static func frames(forSamples samples: Int) -> Int { (samples + hop - 1) / hop }

    /// `wav`: (B, samples) mono at 48 kHz. Returns the VAE mean, (B, frames, 128). Reflect-pads to a multiple of the hop.
    public func encode(_ wav: MLXArray) -> MLXArray {
        var x = wav.expandedDimensions(axis: -1)
        let rem = wav.dim(1) % Self.hop
        if rem != 0 {
            let p = Self.hop - rem, n = x.dim(1)
            let mirror = MLXArray((1...p).map { Int32(n - 1 - $0) })  // x[n-2], x[n-3], … (torch "reflect")
            x = concatenated([x, x.take(mirror, axis: 1)], axis: 1)
        }
        var h = encIn(x)
        for blk in encBlocks {
            for u in blk.units { h = u(h) }
            h = blk.down(snake(h, blk.alpha))
        }
        h = inProj(encOut(snake(h, encAlpha)))
        return h[0..., 0..., 0..<Self.latentDim]
    }

    /// The decoder blocks' output, (B, samples, 96): everything before the output conv and the watermark.
    func decodeBlocks(_ latent: MLXArray) -> MLXArray {
        var h = decIn(outProj(latent))
        for blk in decBlocks {
            h = blk.up(snake(h, blk.alpha))
            for u in blk.units {
                h = u(h)
                eval(h)
            }
        }
        return h
    }

    /// The unmarked waveform from the blocks' output: tanh(conv(snake(x))). (B, samples, 1).
    func plain(fromBlocks x: MLXArray) -> MLXArray { tanh(outConv(snake(x, outAlpha))) }

    /// The watermark signal for an unmarked waveform. `message`: (B, 16) of 0 / 1.
    func mark(_ plain: MLXArray, message: MLXArray) -> MLXArray {
        var h = markIn(plain)
        for g in markShrink.reversed() {
            h = g.conv(elu(g.unit(h)))
            eval(h)
        }
        h = markEncOut(elu(markEncLSTM(h)))
        let idx = MLXArray((0..<Self.messageBits).map { Int32(2 * $0) }).expandedDimensions(axis: 0) + message.asType(.int32)
        h = h + messageTable[idx].sum(axis: 1).expandedDimensions(axis: 1).asType(h.dtype)
        h = markDecLSTM(markDecIn(h))
        for g in markGrow {
            h = g.unit(g.up(elu(h)))
            eval(h)
        }
        return markOut(elu(h))
    }

    /// The unmarked waveform, decoded in windows of `core` latent frames with `margin` frames of context on each side, so the
    /// decoder's 48 kHz activations stay bounded (a whole 10 s clip in one pass peaks near 17 GB). The main path is
    /// non-causal with a receptive field under 8 frames a side (input conv 3, the first block's transposed conv 1 and its
    /// dilated units 3.25, the later blocks 0.5 together), so with 12 frames of margin every kept sample sees exactly the
    /// context it sees in a single pass. (B, samples, 1).
    func plainWindowed(_ latent: MLXArray, core: Int = 25, margin: Int = 12) -> MLXArray {
        let frames = latent.dim(1)
        if frames <= core + 2 * margin { return plain(fromBlocks: decodeBlocks(latent)) }
        var parts = [MLXArray]()
        var start = 0
        while start < frames {
            let end = min(frames, start + core), a = max(0, start - margin), b = min(frames, end + margin)
            let w = plain(fromBlocks: decodeBlocks(latent[0..., a..<b, 0...]))
            let piece = w[0..., ((start - a) * Self.hop)..<((end - a) * Self.hop), 0...]
            eval(piece)
            parts.append(piece)
            MLX.Memory.clearCache()
            start = end
        }
        return concatenated(parts, axis: 1)
    }

    /// Parity lane only: the waveform without the watermark. (B, samples).
    func decodePlain(_ latent: MLXArray) -> MLXArray { plain(fromBlocks: decodeBlocks(latent)).squeezed(axis: -1) }

    /// `latent`: (B, frames, 128). Returns the watermarked waveform, (B, frames · 1920). `message`: (B, 16) of 0 / 1; random
    /// per decode when nil, as upstream.
    public func decode(_ latent: MLXArray, message: MLXArray? = nil) -> MLXArray {
        let p = plainWindowed(latent)
        eval(p)
        let msg = message ?? MLXRandom.randInt(low: 0, high: 2, [latent.dim(0), Self.messageBits])
        return (p + Self.alpha * mark(p, message: msg)).squeezed(axis: -1)
    }
}

extension DACVAE {
    /// Lanes: the blocks' output, the plain waveform and the marked one from the same blocks pass.
    /// Lane: the windowed plain waveform (the production path) for comparison with the single-pass one.
    public func parityPlainWindowed(_ latent: MLXArray) -> MLXArray { plainWindowed(latent).squeezed(axis: -1) }

    public func parityOutputs(_ latent: MLXArray, message: MLXArray) -> (blocks: MLXArray, plain: MLXArray, marked: MLXArray) {
        let b = decodeBlocks(latent)
        let p = plain(fromBlocks: b)
        eval(p)
        return (b, p.squeezed(axis: -1), (p + Self.alpha * mark(p, message: message)).squeezed(axis: -1))
    }
}
