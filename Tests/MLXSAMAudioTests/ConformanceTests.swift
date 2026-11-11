// ConformanceTests.swift — the offline conformance gate for MLXSAMAudio: manifest (C0 / C1 / C7 / C8 / C-memory / C11 /
// provenance; the SAM License admitted on both layers with its port-code notice), the MAT gate, the CAN gate (pre-cancelled run
// propagation, checkpoint cadence), the unloaded and unoffered-tier refusals, the request plane (description → prompt, spans,
// stems, seed; everything undeclared refused), and the audio helpers. Nothing here evals a kernel: the live run is
// `samaudio-gates validate` (GPU, through MLXServeEngine).

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXSAMAudio
@testable import SAMAudioCore

private func wav(seconds: Double = 1, sampleRate: Int = 48_000) -> Audio {
    let n = Int(seconds * Double(sampleRate))
    let samples = (0..<n).map { Float(sin(Double($0) * 2 * .pi * 440 / Double(sampleRate))) * 0.25 }
    return Audio(format: .wav, data: SAMAudioAudioIO.encodeWAV16(samples: samples, sampleRate: sampleRate), sampleRate: sampleRate, channels: 1)
}

final class ManifestConformanceTests: XCTestCase {

    /// C7 / C8 — SAM License on the weights AND the port code (a translation of SAM-licensed code); admitted by the default
    /// policy; the port-code licence carries its notice (contract 1.62.0); the watermark is stated.
    func testLicenseIsSAMOnBothLayersWithItsNotice() {
        let license = SAMAudioPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .sam)
        XCTAssertEqual(license.additionalWeightLicenses, [.apache2])
        XCTAssertEqual(license.portCodeLicense, .sam)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
        XCTAssertNil(license.portCodeLicenseLackingNotice)
        XCTAssertTrue(license.missingRequiredDisplayNotices.isEmpty)
        let notices = license.notices ?? []
        XCTAssertEqual(notices.count, 3)
        XCTAssertTrue(notices.allSatisfy { $0.display == .informational })
        XCTAssertEqual(notices.first?.license, .sam)
        let text = notices.map(\.text).joined(separator: " ")
        for phrase in ["SAM License", "same agreement", "military", "watermark", "not sample-aligned", "guitar", "under 5 s"] {
            XCTAssertTrue(text.contains(phrase), phrase)
        }
    }

    func testEngineGateAdmitsThePackage() async throws {
        for engine in [MLXServeEngine(), MLXServeEngine(licenseEnforcement: .blocking)] {
            _ = try await engine.register(SAMAudioPackage.registration, configuration: SAMAudioConfiguration(), id: PackageID("sam-audio-base"))
            let advisories = await engine.licenseAdvisories
            XCTAssertTrue(advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
        }
    }

    /// C-memory — one tier, split footprint, at least what `samaudio-gates validate` measured (3.38 GB resident; 9.07 GB peak
    /// with the 30 s envelope — the larger of two runs).
    func testFootprintIsTheMeasuredOne() {
        let footprints = SAMAudioPackage.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.fp16])
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 3_380_000_000)
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes + footprints[0].peakActivationBytes, 9_070_000_000)
        XCTAssertEqual(SAMAudioPackage.offeredQuants, [.fp16])
        XCTAssertEqual(SAMAudioConfiguration().quant, .fp16)
        XCTAssertTrue(CancellationConformance.longRunImplied(by: SAMAudioPackage.manifest))
    }

    /// C1 / C11 — `audioSeparation` only; a described target with spans, target and residual, nothing else.
    func testCapabilityIsDescribedSeparation() {
        XCTAssertEqual(Set(SAMAudioPackage.manifest.capabilities), [.audioSeparation])
        let surface = SAMAudioPackage.manifest.surfaces.first { $0.capability == .audioSeparation }
        XCTAssertEqual(surface?.name, "sam-audio-base")
        let controls = SAMAudioPackage.controls
        XCTAssertEqual(controls.stems, [.target, .residual])
        XCTAssertTrue(controls.fixedStems.isEmpty)
        XCTAssertTrue(controls.speakerSelection.isEmpty)
        XCTAssertEqual(controls.describedTarget?.supportsSpans, true)
        XCTAssertTrue(controls.describedTarget?.phrasing?.contains("lowercase") ?? false)
        let names = Set(surface?.parameters.map(\.name) ?? [])
        XCTAssertTrue(names.isSuperset(of: ["audio", "targetDescription", "spans"]), "\(names)")
        XCTAssertFalse(names.contains("speaker"))
        XCTAssertTrue(surface?.parameters.first { $0.name == "targetDescription" }?.required ?? false)
        XCTAssertTrue(surface?.summary.contains("SAM License") ?? false)
        XCTAssertTrue(surface?.summary.contains("watermarked") ?? false)
    }

    /// C0 — the manifest targets a contract with the described target (1.62.0 or later).
    func testContractVersionHasTheDescribedTarget() {
        let v = SAMAudioPackage.manifest.contractVersion
        XCTAssertTrue(v.major > 1 || (v.major == 1 && v.minor >= 62), "\(v)")
    }

    func testProvenancePinsTheUpstreamCheckpoint() {
        let p = SAMAudioPackage.manifest.provenance
        XCTAssertEqual(p.sourceRepo, "facebook/sam-audio-base")
        XCTAssertEqual(p.revision, "81f64008f9f957c2b57a45923fb5299c66e9d186")
        XCTAssertEqual(SAMAudioConfiguration.revisionPin.count, 40)
    }
}

final class MaterializationConformanceTests: XCTestCase {
    private func emptyStoreRoot() -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("samaudio-empty-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func satisfiedConfiguration() throws -> (SAMAudioConfiguration, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("samaudio-bundle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data().write(to: dir.appendingPathComponent(SAMAudioConfiguration.probe))
        return (SAMAudioConfiguration(bundleDirectory: dir), dir)
    }

    /// MAT-1..5: a fresh (dir-less) configuration reports the source missing; an explicit bundle satisfies.
    func testMaterializationGate() throws {
        let (satisfied, dir) = try satisfiedConfiguration()
        defer { try? FileManager.default.removeItem(at: dir) }
        let report = MaterializationConformance.check(freshConfiguration: SAMAudioConfiguration(modelsRootDirectory: emptyStoreRoot()),
                                                      satisfiedConfiguration: satisfied)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourceIsThePublishedBundleAtItsPin() {
        let sources = SAMAudioConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["bundle"])
        XCTAssertEqual(sources.first?.repo, "mlx-community/sam-audio-base-swift-fp16")
        XCTAssertEqual(sources.first?.revision, "2cf624a4e250db577661572f07f56b758f46dffd")
        XCTAssertEqual(SAMAudioConfiguration(revision: "abc").weightSources.first?.revision, "abc")
        // namespace we control (fleet weight-durability rule)
        XCTAssertTrue(SAMAudioConfiguration.repo.hasPrefix("mlx-community/"))
    }

    /// Every file the pipeline opens is downloaded, and so is the licence that must travel with the weights.
    func testDeclaredFilesMatchTheBundleLayout() {
        let layout = ["LICENSE", "README.md", "codec.safetensors", "config.json", "model.safetensors", "special_tokens_map.json",
                      "spiece.model", "text_encoder.safetensors", "tokenizer.json", "tokenizer_config.json", ".gitattributes"]
        for file in layout where !["README.md", ".gitattributes"].contains(file) {
            XCTAssertTrue(SAMAudioConfiguration.files.contains { WeightSourceProbe.matches(path: file, glob: $0) }, "not downloaded: \(file)")
        }
    }

    func testExplicitBundleProbe() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("samaudio-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = SAMAudioConfiguration(bundleDirectory: dir)
        XCTAssertEqual(c.missingWeightSources(storeRoot: nil).count, 1)
        try Data().write(to: dir.appendingPathComponent(SAMAudioConfiguration.probe))
        XCTAssertTrue(c.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(c.resolvedBundle(storeRoot: nil), dir)
        XCTAssertEqual(c.prewarmPaths, [dir])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = SAMAudioConfiguration(revision: "r", bundleDirectory: URL(fileURLWithPath: "/tmp/x"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"))
        let json = String(decoding: try JSONEncoder().encode(c), as: UTF8.self)
        XCTAssertFalse(json.contains("/tmp/"))
        let back = try JSONDecoder().decode(SAMAudioConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(back.revision, "r")
        XCTAssertNil(back.bundleDirectory)
        XCTAssertEqual(back.quant, .fp16)
    }
}

final class CancellationConformanceTests: XCTestCase {
    /// CAN-1 / CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = SAMAudioPackage(configuration: SAMAudioConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: AudioSeparationRequest(audio: wav(), targetDescription: "speech"))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every solver step (denoise) and at each of the two decodes, on the RunProgress seams.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: SAMAudioPackage.manifest,
            posture: .cadence([.init(phase: .denoise, unit: .step, reportsRunProgress: true),
                               .init(phase: .decode, unit: .chunk, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {
    func testUnloadedPackageRejectsLegibly() async {
        let package = SAMAudioPackage(configuration: SAMAudioConfiguration())
        do {
            _ = try await package.run(AudioSeparationRequest(audio: wav(), targetDescription: "speech"))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch { XCTFail("expected PackageError, got \(error)") }
    }

    func testUnofferedTierIsRefusedByLoad() async {
        for q in [Quant.int4, .int8, .fp32, .bf16] {
            let package = SAMAudioPackage(configuration: SAMAudioConfiguration(quant: q))
            do {
                try await package.load()
                XCTFail("loaded \(q)")
            } catch let error as SAMAudioPackageError {
                guard case .unsupportedTier = error else { return XCTFail("\(error)") }
            } catch { XCTFail("\(error)") }
        }
    }

    /// The description is trimmed and lower-cased for the model; spans map in time order; stems default to both; the seed
    /// comes from metaData.
    func testRequestMapsOntoThePipeline() throws {
        let req = AudioSeparationRequest(audio: wav(), targetDescription: "  Dog Barking ",
                                         spans: [AudioSeparationSpan(startSeconds: 3, endSeconds: 4, present: false), AudioSeparationSpan(startSeconds: 0.5, endSeconds: 1)],
                                         metaData: ["seed": .int(42)])
        let plan = try SAMAudioPackage.plan(req)
        XCTAssertEqual(plan.prompt, "dog barking")
        XCTAssertEqual(plan.spans, [SeparationSpan(start: 0.5, end: 1, present: true), SeparationSpan(start: 3, end: 4, present: false)])
        XCTAssertEqual(plan.stems, [.target, .residual])
        XCTAssertEqual(plan.seed, 42)
        XCTAssertEqual(try SAMAudioPackage.plan(AudioSeparationRequest(audio: wav(), stems: [.residual], targetDescription: "rain")).stems, [.residual])
    }

    func testUndeclaredFeaturesAndBoundsAreRefused() {
        let bad: [AudioSeparationRequest] = [
            AudioSeparationRequest(audio: wav()),
            AudioSeparationRequest(audio: wav(), targetDescription: "  "),
            AudioSeparationRequest(audio: wav(), stems: [.vocals], targetDescription: "speech"),
            AudioSeparationRequest(audio: wav(), stems: [.target]),
            AudioSeparationRequest(audio: wav(), speaker: .ordinal(1)),
            AudioSeparationRequest(audio: wav(), speaker: .ordinal(1), targetDescription: "speech"),
            AudioSeparationRequest(audio: wav(), spans: [AudioSeparationSpan(startSeconds: 0, endSeconds: 1)]),
            AudioSeparationRequest(audio: wav(), targetDescription: "speech", spans: [AudioSeparationSpan(startSeconds: 2, endSeconds: 1)]),
            AudioSeparationRequest(audio: wav(), targetDescription: String(repeating: "a", count: 201)),
        ]
        for (i, req) in bad.enumerated() {
            XCTAssertThrowsError(try SAMAudioPackage.plan(req), "request \(i)") { error in
                guard case PackageError.unsupportedRequestFeature = error else { return XCTFail("request \(i): \(error)") }
            }
        }
        // the span-past-the-end rule the package applies once it knows the length
        let past = AudioSeparationRequest(audio: wav(), targetDescription: "speech", spans: [AudioSeparationSpan(startSeconds: 1, endSeconds: 12)])
        XCTAssertNil(SAMAudioPackage.controls.refusal(for: past))
        XCTAssertNotNil(SAMAudioPackage.controls.refusal(for: past, audioDurationSeconds: 10))
    }

    /// Anchors as upstream builds them: ids [null, pad, one per span], frames inside a span point at its id (25 frames per second).
    func testAnchorsFollowUpstream() {
        let a = SAMAudioPipeline.anchors([SeparationSpan(start: 0.5, end: 1.9), SeparationSpan(start: 3.0, end: 4.4, present: false)], frames: 250)
        XCTAssertEqual(a.ids.asArray(Int32.self), [0, 3, 1, 2])
        let alignment = a.alignment.asArray(Int32.self)
        XCTAssertEqual(alignment[12], 0); XCTAssertEqual(alignment[13], 2); XCTAssertEqual(alignment[47], 2); XCTAssertEqual(alignment[48], 0)
        // 4.4 s × 48 000 is 211 200.00000000003 in floating point, and upstream takes its ceiling, so the span runs one frame
        // past 110 — kept, because the `e2e anchors` lane compares these arrays with upstream's.
        XCTAssertEqual(alignment[75], 3); XCTAssertEqual(alignment[110], 3); XCTAssertEqual(alignment[111], 0)
    }
}

final class AudioIOTests: XCTestCase {
    func testWAVRoundTripKeepsEveryFrame() throws {
        // 480 000 frames: the length a one-call AVAudioFile read returned short on
        let a = wav(seconds: 10)
        let samples = try SAMAudioAudioIO.decodeMono(a, sampleRate: 48_000)
        XCTAssertEqual(samples.count, 480_000)
        XCTAssertEqual(samples[100], Float(sin(Double(100) * 2 * .pi * 440 / 48_000)) * 0.25, accuracy: 1e-3)
    }

    func testOtherRatesAreConvertedTo48k() throws {
        let samples = try SAMAudioAudioIO.decodeMono(wav(seconds: 2, sampleRate: 44_100), sampleRate: 48_000)
        XCTAssertEqual(Double(samples.count), 96_000, accuracy: 64)
        XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.2)
    }

    func testUnreadableAudioIsRefused() {
        XCTAssertThrowsError(try SAMAudioAudioIO.decodeMono(Audio(format: .wav, data: Data([1, 2, 3])), sampleRate: 48_000)) { error in
            guard case PackageError.unsupportedRequestFeature = error else { return XCTFail("\(error)") }
        }
    }
}
