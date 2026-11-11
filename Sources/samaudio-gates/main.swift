// samaudio-gates — parity lanes against the upstream PyTorch goldens (Tools/oracle-capture; WIP/sam-audio-port/oracle).
//
//   samaudio-gates <lane> --bundle <dir> --goldens <dir of case dirs> [--cases A01_rain,B01_song,…]
//   lanes: keys | t5 | velocity | solve | codec | e2e | all
//   samaudio-gates batch --bundle <dir> --manifest <tsv: wav, prompt, seed, out prefix[, spans a-b;c-d]> [--dtype stored|fp32|fp16] [--steps 16]
//
// Lanes run at fp32 with TF32 off (MLX's default TF32 matmul costs ~55 dB on these lanes; AB-L-0315): the tool sets
// MLX_ENABLE_TF32=0 before its first MLX op and reports what latched.
import Darwin
import Foundation
import MLX
import MLXNN
import SAMAudioCore

TF32.disableForProcess()

var args = Array(CommandLine.arguments.dropFirst())
guard let lane = args.first else { print("usage: samaudio-gates <keys|t5|velocity|solve|all> --bundle <dir> --goldens <dir> [--cases a,b]"); exit(2) }
args.removeFirst()
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
guard let bundlePath = option("--bundle") else { print("--bundle is required"); exit(2) }
let bundle = URL(fileURLWithPath: bundlePath), goldens = URL(fileURLWithPath: option("--goldens") ?? "/nonexistent")

/// `phys_footprint` now and its lifetime peak, GB (the fleet's footprint figure).
func phys() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e9, Double(info.ledger_phys_footprint_peak) / 1e9)
}

if lane == "batch" {
    if let mb = option("--cache-mb").flatMap(Int.init) { MLX.Memory.cacheLimit = mb * 1_048_576 }
    guard let manifestPath = option("--manifest") else { print("--manifest is required"); exit(2) }
    let dtype: DType? = ["fp16": DType.float16, "fp32": DType.float32][option("--dtype") ?? "stored"]  // default: as stored in the bundle
    let steps = Int(option("--steps") ?? "16") ?? 16
    let t0 = Date()
    let pipe = try await SAMAudioPipeline(bundle: bundle, modelDType: dtype)
    print("loaded \(dtype.map { "\($0)" } ?? "as stored") in \(String(format: "%.2f", Date().timeIntervalSince(t0))) s; phys \(String(format: "%.2f", phys().now)) GB; TF32 off: \(TF32.environmentDisables)")
    var times = [Double]()
    for (n, line) in try String(contentsOfFile: manifestPath, encoding: .utf8).split(separator: "\n").enumerated() {
        let f = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        guard f.count >= 4 else { continue }
        let (samples, rate) = try WAV.read(URL(fileURLWithPath: f[0]))
        precondition(rate == DACVAE.sampleRate, "\(f[0]): \(rate) Hz, expected 48000")
        let spans: [SeparationSpan] = f.count > 4 && !f[4].isEmpty ? f[4].split(separator: ";").map { p in
            let ab = p.split(separator: "-").map { Double($0)! }
            return SeparationSpan(start: ab[0], end: ab[1])
        } : []
        let start = Date()
        let out = try pipe.separate(wav: MLXArray(samples), prompt: f[1], spans: spans, seed: UInt64(f[2])!, steps: steps)
        let seconds = Date().timeIntervalSince(start)
        if n > 0 { times.append(seconds) }
        try WAV.write(out.target.asArray(Float.self), sampleRate: DACVAE.sampleRate, to: URL(fileURLWithPath: f[3] + "_target.wav"))
        try WAV.write(out.residual.asArray(Float.self), sampleRate: DACVAE.sampleRate, to: URL(fileURLWithPath: f[3] + "_residual.wav"))
        let p = phys()
        print("\(n)\t\(String(format: "%.2f", seconds)) s\tphys \(String(format: "%.2f", p.now)) / peak \(String(format: "%.2f", p.peak)) GB\t\(f[1])\t\(URL(fileURLWithPath: f[3]).lastPathComponent)")
        MLX.Memory.clearCache()
    }
    let sorted = times.sorted()
    if !sorted.isEmpty { print("median \(String(format: "%.2f", sorted[sorted.count / 2])) s over \(sorted.count) (first excluded); phys peak \(String(format: "%.2f", phys().peak)) GB") }
    exit(0)
}
let cases = (option("--cases") ?? "A01_rain,B01_song,C02_trio,A09_cat").split(separator: ",").map(String.init)

func npy(_ c: String, _ n: String) throws -> MLXArray { try NPY.load(goldens.appendingPathComponent(c).appendingPathComponent(n + ".npy")) }
func relL2(_ a: MLXArray, _ b: MLXArray) -> Double {
    let x = a.asType(.float32), y = b.asType(.float32)
    precondition(x.shape == y.shape, "shape \(x.shape) vs \(y.shape)")
    let num = sqrt(sum((x - y) * (x - y))).item(Float.self), den = sqrt(sum(y * y)).item(Float.self)
    return Double(num) / Double(max(den, 1e-30))
}
func snr(_ r: Double) -> Double { -20 * log10(max(r, 1e-30)) }
var results: [(String, Bool, String)] = []
/// Pass when the lane reads at least `floor` dB; written as !(x >= floor) == fail so a NaN fails (AB-L-0321).
func record(_ lane: String, _ db: Double, floor: Double) {
    let pass = db >= floor
    results.append((lane, pass, String(format: "%.1f dB (floor %.0f)", db, floor)))
    print("\(pass ? "PASS" : "FAIL") \(lane): \(String(format: "%.1f dB", db)) (floor \(Int(floor)))")
}

if let mb = option("--cache-mb").flatMap(Int.init) { MLX.Memory.cacheLimit = mb * 1_048_576 }
let config = try SAMAudioConfig.load(bundle.appendingPathComponent("config.json"))
print("TF32 off: \(TF32.isOff()) (probe relL2 \(String(format: "%.2e", TF32.probeRelL2())))")

func conditioning(_ c: String) throws -> Conditioning {
    Conditioning(audioFeatures: try npy(c, "audio_features"), textFeatures: try npy(c, "text_features"), textMask: try npy(c, "text_mask"),
                 anchorIDs: try npy(c, "anchor_ids"), anchorAlignment: try npy(c, "anchor_alignment"), audioPadMask: try npy(c, "audio_pad_mask"))
}

if lane == "keys" || lane == "all" {
    // S0: the modules' parameter paths against the bundle's keys, weight-free.
    for (file, module) in [("model.safetensors", SAMAudioModel(config) as Module), ("text_encoder.safetensors", T5Encoder(config.text_encoder) as Module)] {
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(try loadArrays(url: bundle.appendingPathComponent(file)).keys)
        let missing = expected.subtracting(onDisk).sorted(), unused = onDisk.subtracting(expected).sorted()
        let pass = missing.isEmpty && unused.isEmpty
        results.append(("keys \(file)", pass, "\(expected.count) keys, missing \(missing.count), unused \(unused.count)"))
        print("\(pass ? "PASS" : "FAIL") keys \(file): \(expected.count) module keys, \(onDisk.count) on disk, missing \(missing.prefix(5)), unused \(unused.prefix(5))")
    }
}

if lane == "t5" || lane == "all" {
    let t5 = try T5Encoder.load(bundle: bundle, config: config, dtype: .float32)
    for c in cases {
        let out = t5(try npy(c, "input_ids"), mask: try npy(c, "text_mask"))
        record("t5 \(c)", snr(relL2(out, try npy(c, "text_features"))), floor: 100)
    }
}

if lane == "velocity" || lane == "solve" || lane == "all" {
    let model = try SAMAudioModel.load(bundle: bundle, config: config, dtype: .float32)
    for c in cases {
        let cond = try conditioning(c)
        let noise = try npy(c, "noise")
        if lane != "solve" {
            let v = model(noisy: noise, time: try npy(c, "t"), cond)
            record("velocity \(c)", snr(relL2(v, try npy(c, "velocity_upstream"))), floor: 100)
        }
        if lane != "velocity" {
            let start = Date()
            let x = FlowSolver.solve(model, noise: noise, cond)
            record("solve \(c)", snr(relL2(x, try npy(c, "latent"))), floor: 100)
            print("      16 midpoint steps in \(String(format: "%.2f", Date().timeIntervalSince(start))) s")
        }
    }
}

if lane == "tier" {
    // A half-precision DiT against the fp32 goldens: one velocity and the 16-step solve. Floors 40 / 30 dB.
    let model = try SAMAudioModel.load(bundle: bundle, config: config, dtype: .float16)
    for c in cases {
        var cond = try conditioning(c)
        cond.audioFeatures = cond.audioFeatures.asType(.float16)
        cond.textFeatures = cond.textFeatures.asType(.float16)
        let noise = try npy(c, "noise").asType(.float16)
        record("fp16 velocity \(c)", snr(relL2(model(noisy: noise, time: try npy(c, "t"), cond), try npy(c, "velocity_upstream"))), floor: 40)
        record("fp16 solve \(c)", snr(relL2(FlowSolver.solve(model, noise: noise, cond), try npy(c, "latent"))), floor: 30)
    }
}

if lane == "codecmem" {
    // Memory of the production decode alone: one 10 s item, windowed plain + watermark.
    let codec = try DACVAE.load(bundle: bundle, dtype: .float32)
    for c in cases {
        let latent = try npy(c, "decode_in").transposed(0, 2, 1)[0..<1]
        let t0 = Date()
        let w = codec.decode(latent, message: try npy(c, "message"))
        eval(w)
        print("codecmem \(c): decode + watermark of 10 s in \(String(format: "%.2f", Date().timeIntervalSince(t0))) s; vs golden marked \(String(format: "%.1f", snr(relL2(w, try npy(c, "marked").squeezed(axis: 1)[0..<1])))) dB; phys peak \(String(format: "%.2f", phys().peak)) GB")
    }
    exit(0)
}

if lane == "codec" || lane == "all" {
    // Floor 90 dB: the Python-MLX rung's own worst codec case against upstream is 94.5 dB (encode, the cat mixture).
    let start = Date()
    let codec = try DACVAE.load(bundle: bundle, dtype: .float32)
    print("codec loaded in \(String(format: "%.2f", Date().timeIntervalSince(start))) s")
    for c in cases {
        let feats = codec.encode(try npy(c, "wav").expandedDimensions(axis: 0))
        let golden = try npy(c, "audio_features")
        record("codec encode \(c)", snr(relL2(feats, golden[0..., 0..., 0..<DACVAE.latentDim])), floor: 90)
        print("      phys peak after encode \(String(format: "%.2f", phys().peak)) GB")
        let latent = try npy(c, "decode_in").transposed(0, 2, 1)  // (2, 128, T) -> (2, T, 128): target and residual
        let message = repeated(try npy(c, "message"), count: latent.dim(0), axis: 0)
        let t0 = Date()
        let out = codec.parityOutputs(latent, message: message)
        eval(out.marked)
        let seconds = Date().timeIntervalSince(t0)
        record("codec blocks \(c)", snr(relL2(out.blocks, try npy(c, "blocks").transposed(0, 2, 1))), floor: 90)
        let plain = try npy(c, "plain").squeezed(axis: 1), marked = try npy(c, "marked").squeezed(axis: 1)
        record("codec plain \(c)", snr(relL2(out.plain, plain)), floor: 90)
        record("codec marked \(c)", snr(relL2(out.marked, marked)), floor: 90)
        // S5: the windowed decode (production) against the golden and against the single pass
        let windowed = codec.parityPlainWindowed(latent)
        record("codec plain windowed \(c)", snr(relL2(windowed, plain)), floor: 90)
        record("codec windowed vs single pass \(c)", snr(relL2(windowed, out.plain)), floor: 100)
        // the watermark itself, as a signal: (marked − plain) against the golden's
        record("codec mark signal \(c)", snr(relL2(out.marked - out.plain, marked - plain)), floor: 60)
        print("      decode + watermark of 2 × 10 s in \(String(format: "%.2f", seconds)) s; phys peak \(String(format: "%.2f", phys().peak)) GB")
    }
}

if lane == "e2e" || lane == "all" {
    // wav + prompt (+ spans) through the whole Swift pipeline, the golden noise and the fixed watermark message.
    let pipe = try await SAMAudioPipeline(bundle: bundle, modelDType: .float32)
    for c in cases {
        let prompt = try String(contentsOf: goldens.appendingPathComponent(c).appendingPathComponent("prompt.txt"), encoding: .utf8)
        let ids = pipe.tokens(prompt), goldenIDs = try npy(c, "input_ids").asArray(Int32.self)
        let tokensMatch = ids == goldenIDs
        results.append(("e2e tokens \(c)", tokensMatch, "\(ids)"))
        print("\(tokensMatch ? "PASS" : "FAIL") e2e tokens \(c): \(ids) vs \(goldenIDs)")
        let spans: [SeparationSpan] = c == "A09_cat" ? [(0.5, 1.9), (3.0, 4.4), (5.5, 6.9), (7.95, 9.35)].map { SeparationSpan(start: $0.0, end: $0.1) } : []
        let wav = try npy(c, "wav")
        let cond = pipe.conditioning(wav: wav, prompt: prompt, spans: spans)
        let goldenAnchorIDs = try npy(c, "anchor_ids").asArray(Int32.self), goldenAlignment = try npy(c, "anchor_alignment").asArray(Int32.self)
        let anchorsMatch = cond.anchorIDs.asArray(Int32.self) == goldenAnchorIDs && cond.anchorAlignment.asArray(Int32.self) == goldenAlignment
        results.append(("e2e anchors \(c)", anchorsMatch, ""))
        print("\(anchorsMatch ? "PASS" : "FAIL") e2e anchors \(c)")
        let t0 = Date()
        let out = try pipe.separate(wav: wav, prompt: prompt, spans: spans, noise: try npy(c, "noise"), message: try npy(c, "message"))
        let seconds = Date().timeIntervalSince(t0)
        let marked = try npy(c, "marked").squeezed(axis: 1)
        record("e2e target \(c)", snr(relL2(out.target, marked[0])), floor: 90)
        record("e2e residual \(c)", snr(relL2(out.residual, marked[1])), floor: 90)
        print("      10 s clip end to end in \(String(format: "%.2f", seconds)) s (fp32); phys peak \(String(format: "%.2f", phys().peak)) GB")
    }
}

let failed = results.filter { !$0.1 }
print("\n\(results.count - failed.count)/\(results.count) lanes pass" + (failed.isEmpty ? "" : " — FAILED: " + failed.map(\.0).joined(separator: ", ")))
exit(failed.isEmpty ? 0 : 1)
