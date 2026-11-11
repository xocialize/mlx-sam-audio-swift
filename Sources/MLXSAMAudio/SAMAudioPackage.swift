import Foundation
import MLX
import MLXToolKit
import SAMAudioCore

/// Meta's SAM-Audio base on the canonical `audioSeparation` surface (contract ≥ 1.62.0): a mixture + a described target (+ time
/// spans, + a seed in `metaData`) → the described sound (`Stem.target`) and everything else (`Stem.residual`), each a 48 kHz
/// mono `.wav` of the mixture's length (E29, AB-T-0470; operator approval AB-D-0221).
///
/// Engine-owned lifecycle (C13): the engine constructs from a `SAMAudioConfiguration`, materializes the declared bundle into its
/// store (or the host names one), pages it in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// What the surface declares (`AudioSeparationControls`): a described target with spans, and the two stems it makes. No fixed
/// stems and no speaker picks, so the engine refuses a request without a description before admission. The stems are GENERATED
/// (a flow model decoding through a codec), not filtered: they are not sample-aligned with the mixture, and nothing may be
/// derived from them by subtraction.
///
/// Every output carries the codec's AudioSeal watermark, as upstream's does (operator decision AB-D-0222); there is no switch.
///
/// Licence: weights AND port code are under Meta's SAM License (`SPDXLicense.sam`, allowlisted; AB-D-0220, AB-D-0223) — the
/// code is a translation of SAM-licensed code. The T5-base encoder inside the bundle is Apache-2.0.
///
/// Sampling: the flow starts from noise drawn from MLX's generator keyed with the seed (`metaData["seed"]`, an int; random
/// when absent), and the watermark's message is drawn from the same seed, so a seed reproduces a separation to the sample
/// within Swift-MLX.
@InferenceActor
public final class SAMAudioPackage: ModelPackage {
    public typealias Configuration = SAMAudioConfiguration

    /// Footprints MEASURED as phys_footprint, pool-inclusive (AB-L-0113 / AB-L-0155), MLX cache 1 GiB, through MLXServeEngine
    /// (`samaudio-gates validate`, M5 Max, 2026-11-11; MEASUREMENTS.md §V3): resident = phys after `prepare`, 3.36 / 3.38 GB
    /// over two runs; activation = the lifetime peak over the validate requests minus the resident — 9.01 / 9.07 GB with the 30 s
    /// mixture (the envelope; 5.66 / 5.68 GB of activation), 6.28 / 6.45 GB when no request is over 10 s (2.92 / 3.06 GB). Declared rounded up to 0.5 GB:
    /// 3.5 + 6.0. By the engine's 0.7 × RAM admission that is a 16 GB-Mac tier (9.5 ≤ 11.2).
    nonisolated static let fp16ResidentBytes: UInt64 = 3_500_000_000
    nonisolated static let fp16PeakActivationBytes: UInt64 = 6_000_000_000

    public nonisolated static let sampleRate = DACVAE.sampleRate
    /// The longest mixture taken. Gated at 5–30 s (median gain 10.4–12.1 dB, no failures; WIP/sam-audio-eval length sweep).
    public nonisolated static let maxDurationSeconds: Double = 30
    /// The shortest mixture taken. Under 5 s the model is unreliable (2 s: 2 of 8 separations worse than the mixture); the
    /// surface says so and still runs down to 1 s.
    public nonisolated static let minDurationSeconds: Double = 1
    public nonisolated static let maxDescriptionCharacters = 200
    public nonisolated static let solverSteps = 16

    public nonisolated static let phrasing =
        "Name the sound in a few lowercase words — \"speech\", \"dog barking\", \"drums\". Sentences do not work."

    public nonisolated static var controls: AudioSeparationControls {
        AudioSeparationControls(stems: [.target, .residual],
                                describedTarget: DescribedTargetControls(supportsSpans: true, phrasing: phrasing))
    }

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights are under the SAM License; the T5-base encoder in the bundle is Apache-2.0. C8: the port code is a
            // translation of SAM-licensed code, so it is under the SAM License too and owes the notice below (contract 1.62.0).
            license: LicenseDeclaration(
                weightLicense: .sam, additionalWeightLicenses: [.apache2], portCodeLicense: .sam,
                notices: [
                    LicenseNotice(text: "Model and code are under Meta's SAM License: commercial use is allowed; redistribute only under "
                                  + "the same agreement with a copy of it; no military, warfare, nuclear, espionage or weapons end uses.",
                                  display: .informational, license: .sam,
                                  termsURL: "https://huggingface.co/facebook/sam-audio-base/blob/main/LICENSE"),
                    LicenseNotice(text: "Every output carries an AudioSeal watermark, as upstream's does. It cannot be turned off.",
                                  display: .informational),
                    LicenseNotice(text: "Stems are generated, not filtered: they are not sample-aligned with the mixture, so nothing may be "
                                  + "derived from them by subtraction. Works on general sounds, speech against music, drums and bass; "
                                  + "guitar-like instruments do not separate. Mixtures under 5 s are unreliable.",
                                  display: .informational),
                ]),
            provenance: Provenance(sourceRepo: SAMAudioConfiguration.upstreamRepo, revision: SAMAudioConfiguration.upstreamRevision, tier: 3),
            requirements: RequirementsManifest(
                footprints: [QuantFootprint(quant: .fp16, residentBytes: fp16ResidentBytes, peakActivationBytes: fp16PeakActivationBytes)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                AudioSeparationContract.descriptor(
                    name: "sam-audio-base",
                    summary: "SAM-Audio base (Meta, SAM License): separates the sound you describe from a mixture of 1–30 s — speech, a "
                        + "sound effect, music, drums, bass — and returns it (target) with everything else (residual), 48 kHz mono, "
                        + "generated and watermarked. Optional time spans say where the sound is, or is not, present; they help "
                        + "intermittent sounds. metaData: seed.",
                    controls: Self.controls
                )
            ]
        )
    }

    /// The only tier offered: the published fp16 bundle.
    public nonisolated static let offeredQuants: [Quant] = [.fp16]

    private let configuration: Configuration
    private var pipeline: SAMAudioPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard pipeline == nil else { return }
        guard Self.offeredQuants.contains(configuration.quant) else {
            throw SAMAudioPackageError.unsupportedTier("quant \(configuration.quant) — this package offers fp16 only")
        }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw SAMAudioPackageError.missingWeights(
                "source not materialized: \(missing.map(\.repo).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let bundle = configuration.resolvedBundle(storeRoot: storeRoot) else {
            throw SAMAudioPackageError.missingWeights("unresolved bundle directory (no store root)")
        }
        MLX.Memory.cacheLimit = Int(configuration.cacheLimitBytes)
        do {
            // Weight loads ride the CPU stream inside the Core (WeightIO.load) — the Metal-watchdog rule. The DiT is used as
            // stored (fp16); the T5 encoder and the codec are fp32 in the bundle and stay fp32.
            pipeline = try await SAMAudioPipeline(bundle: bundle)
        } catch let error as SAMAudioError {
            throw SAMAudioPackageError.loadFailed(error.description)
        }
        MLX.Memory.clearCache()
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every solver step (denoise) and each of the
        // two decodes (decode), on the same seams as the RunProgress reports; the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .audioSeparation, let req = request as? AudioSeparationRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req)
        let samples = try SAMAudioAudioIO.decodeMono(req.audio, sampleRate: Self.sampleRate)
        let seconds = Double(samples.count) / Double(Self.sampleRate)
        // The one refusal the engine cannot make: a span past the end needs the decoded length (contract 1.62.0).
        if let refusal = Self.controls.refusal(for: req, audioDurationSeconds: seconds) {
            throw PackageError.unsupportedRequestFeature(refusal)
        }
        guard seconds >= Self.minDurationSeconds, seconds <= Self.maxDurationSeconds else {
            throw PackageError.unsupportedRequestFeature(String(
                format: "the mixture is %.2f s — this surface takes %g…%g s", seconds, Self.minDurationSeconds, Self.maxDurationSeconds))
        }
        try Task.checkCancellation()
        let out: SeparationOutput
        do {
            out = try pipeline.separate(
                wav: MLXArray(samples), prompt: plan.prompt, spans: plan.spans, seed: plan.seed, steps: Self.solverSteps,
                onStep: { step in
                    RunProgress.report(.denoise, step: step, totalSteps: Self.solverSteps)
                    try Task.checkCancellation()
                },
                onDecode: { part, parts in
                    try Task.checkCancellation()
                    RunProgress.report(.decode, step: part, totalSteps: parts)
                })
        } catch let error as SAMAudioError {
            throw SAMAudioPackageError.separationFailed(error.description)
        }
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        var stems = [Stem: Audio]()
        for stem in plan.stems {
            let wave = (stem == .target ? out.target : out.residual).asArray(Float.self)
            stems[stem] = Audio(format: .wav, data: SAMAudioAudioIO.encodeWAV16(samples: wave, sampleRate: Self.sampleRate),
                                sampleRate: Self.sampleRate, channels: 1)
        }
        MLX.Memory.clearCache()
        return AudioSeparationResponse(stems: stems)
    }

    /// What the request asks for, judged against the declaration (the engine admits only declared features; a direct caller
    /// meets the same rules here), mapped onto the Core's inputs.
    public struct Plan: Equatable {
        /// The description as the model gets it: trimmed and lower-cased.
        public let prompt: String
        public let spans: [SeparationSpan]
        /// The stems to return, in the request's order (both when it names none).
        public let stems: [Stem]
        public let seed: UInt64
    }

    public nonisolated static func plan(_ req: AudioSeparationRequest) throws -> Plan {
        if let refusal = controls.refusal(for: req) { throw PackageError.unsupportedRequestFeature(refusal) }
        // the declaration makes a description mandatory; this is the direct caller's backstop
        let prompt = (req.targetDescription ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !prompt.isEmpty else {
            throw PackageError.unsupportedRequestFeature("targetDescription — name the sound to separate (\(phrasing))")
        }
        guard prompt.count <= maxDescriptionCharacters else {
            throw PackageError.unsupportedRequestFeature(
                "targetDescription is \(prompt.count) characters — at most \(maxDescriptionCharacters); \(phrasing)")
        }
        let stems: [Stem] = req.stems.isEmpty ? [.target, .residual] : req.stems
        let spans = req.spans.sorted { $0.startSeconds < $1.startSeconds }
            .map { SeparationSpan(start: $0.startSeconds, end: $0.endSeconds, present: $0.present) }
        let seed: UInt64
        if case .int(let v)? = req.metaData["seed"] { seed = UInt64(bitPattern: Int64(v)) } else { seed = UInt64.random(in: 0 ... UInt64(UInt32.max)) }
        return Plan(prompt: prompt, spans: spans, stems: stems, seed: seed)
    }
}

extension SAMAudioPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(SAMAudioPackage.self) }
}

/// Wrapper-level errors (weight resolution, tier, a Core failure). Runtime request errors use `PackageError`.
public enum SAMAudioPackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    case unsupportedTier(String)
    case loadFailed(String)
    case separationFailed(String)
    public var description: String {
        switch self {
        case .missingWeights(let s): return "SAM-Audio: \(s)"
        case .unsupportedTier(let s): return "SAM-Audio: unsupported tier \(s)"
        case .loadFailed(let s): return "SAM-Audio: load failed — \(s)"
        case .separationFailed(let s): return "SAM-Audio: \(s)"
        }
    }
}
