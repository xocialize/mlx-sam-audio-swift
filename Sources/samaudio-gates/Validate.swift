// Validate.swift — the live lane: `MLXSAMAudio` through `MLXServeEngine` (register → materialize / prepare → run → refusals →
// cancel → evict), with `phys_footprint` read at each point. This is where the manifest's footprint constants come from.
//
//   samaudio-gates validate --mixture <≈10 s wav> --long <≈30 s wav> --toolong <> 30 s wav> --out <dir>
//                           [--bundle <dir> | --store <models root>]
import Darwin
import Foundation
import MLX
import MLXSAMAudio
import MLXServeCore
import MLXToolKit
import SAMAudioCore

func physGB() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e9, Double(info.ledger_phys_footprint_peak) / 1e9)
}

func validate(bundle: URL?, store: URL?, mixture: URL, long: URL, tooLong: URL, out: URL) async throws -> Bool {
    var ok = true
    func check(_ name: String, _ pass: Bool, _ detail: String = "") {
        ok = ok && pass
        print("\(pass ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : ": " + detail)")
    }
    func f(_ v: Double) -> String { String(format: "%.2f", v) }
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let engine = MLXServeEngine()   // the default policy (.permissiveOnly, .advisory)
    if let store { await engine.useModelStore(ModelStore(root: store)) }
    let configuration = SAMAudioConfiguration(bundleDirectory: bundle)
    let pid = try await engine.register(SAMAudioPackage.registration, configuration: configuration, id: PackageID("sam-audio-base"))
    let advisories = await engine.licenseAdvisories
    check("registered under the default licence policy with no advisory", advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
    if await engine.needsDownload(.audioSeparation, package: pid) {
        let t = Date()
        _ = try await engine.materialize(package: pid)
        print("materialized in \(f(Date().timeIntervalSince(t))) s")
        check("nothing left to download", !(await engine.needsDownload(.audioSeparation, package: pid)))
    }
    let before = physGB().now
    var t0 = Date()
    try await engine.prepare(.audioSeparation, package: pid)
    let resident = physGB().now
    let charged = Double(await engine.memory.residentBytes) / 1e9
    print("prepare \(f(Date().timeIntervalSince(t0))) s; phys before \(f(before)) GB, after prepare \(f(resident)) GB; engine charges \(f(charged)) GB")

    func audio(_ url: URL) throws -> Audio { Audio(format: .wav, data: try Data(contentsOf: url)) }
    func samples(_ a: Audio) throws -> [Float] {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("v-\(UUID().uuidString).wav")
        try a.data.write(to: tmp); defer { try? FileManager.default.removeItem(at: tmp) }
        return try WAV.read(tmp).samples
    }
    func rms(_ x: [Float]) -> Double { x.isEmpty ? 0 : (x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count)).squareRoot() }
    let mix = try audio(mixture), mixSamples = try WAV.read(mixture).samples
    let longMix = try audio(long), longSamples = try WAV.read(long).samples

    func run(_ label: String, _ request: AudioSeparationRequest, expect: [Stem], length: Int, mixRMS: Double) async throws {
        t0 = Date()
        guard let response = try await engine.run(request, package: pid) as? AudioSeparationResponse else { check(label, false, "no AudioSeparationResponse"); return }
        let seconds = Date().timeIntervalSince(t0)
        var notes = [String]()
        var pass = Set(response.stems.keys) == Set(expect)
        for stem in expect {
            guard let a = response.stems[stem] else { continue }
            let s = try samples(a)
            pass = pass && s.count == length && a.sampleRate == 48_000 && a.channels == 1 && rms(s) > 0.02 * mixRMS && s.allSatisfy(\.isFinite)
            notes.append("\(stem.rawValue) \(s.count) samples rms \(String(format: "%.4f", rms(s)))")
            try a.data.write(to: out.appendingPathComponent("\(label)_\(stem.rawValue).wav"))
        }
        let p = physGB()
        check(label, pass, "\(notes.joined(separator: ", ")); \(f(seconds)) s; phys \(f(p.now)) / peak \(f(p.peak)) GB")
    }
    let mixRMS = rms(mixSamples), longRMS = rms(longSamples)
    try await run("speech", AudioSeparationRequest(audio: mix, targetDescription: "speech", metaData: ["seed": .int(7)]),
                  expect: [.target, .residual], length: mixSamples.count, mixRMS: mixRMS)
    try await run("speech-again", AudioSeparationRequest(audio: mix, targetDescription: "  Speech ", metaData: ["seed": .int(7)]),
                  expect: [.target, .residual], length: mixSamples.count, mixRMS: mixRMS)
    try await run("target-only", AudioSeparationRequest(audio: mix, stems: [.target], targetDescription: "speech", metaData: ["seed": .int(7)]),
                  expect: [.target], length: mixSamples.count, mixRMS: mixRMS)
    try await run("spans", AudioSeparationRequest(audio: mix, targetDescription: "speech",
                                                  spans: [AudioSeparationSpan(startSeconds: 0.5, endSeconds: 4.0), AudioSeparationSpan(startSeconds: 8.0, endSeconds: 9.5, present: false)],
                                                  metaData: ["seed": .int(7)]),
                  expect: [.target, .residual], length: mixSamples.count, mixRMS: mixRMS)
    let afterShort = physGB().peak
    try await run("long-30s", AudioSeparationRequest(audio: longMix, targetDescription: "speech", metaData: ["seed": .int(7)]),
                  expect: [.target, .residual], length: longSamples.count, mixRMS: longRMS)
    let afterLong = physGB().peak
    print("phys peak: ≤ 10 s requests \(f(afterShort)) GB (activation \(f(afterShort - resident))), with the 30 s request \(f(afterLong)) GB (activation \(f(afterLong - resident)))")

    // refusals — each must be refused, legibly
    let seconds = Double(mixSamples.count) / 48_000
    let refusals: [(String, AudioSeparationRequest)] = [
        ("no description", AudioSeparationRequest(audio: mix)),
        ("blank description", AudioSeparationRequest(audio: mix, targetDescription: "   ")),
        ("a fixed stem beside a description", AudioSeparationRequest(audio: mix, stems: [.vocals], targetDescription: "speech")),
        ("a speaker pick", AudioSeparationRequest(audio: mix, speaker: .ordinal(1), targetDescription: "speech")),
        ("span with end before start", AudioSeparationRequest(audio: mix, targetDescription: "speech", spans: [AudioSeparationSpan(startSeconds: 3, endSeconds: 2)])),
        ("span past the mixture's end", AudioSeparationRequest(audio: mix, targetDescription: "speech", spans: [AudioSeparationSpan(startSeconds: 1, endSeconds: seconds + 5)])),
        ("a mixture over 30 s", AudioSeparationRequest(audio: try audio(tooLong), targetDescription: "speech")),
        ("a 201-character description", AudioSeparationRequest(audio: mix, targetDescription: String(repeating: "a", count: 201))),
    ]
    for (label, bad) in refusals {
        do { _ = try await engine.run(bad, package: pid); check("refused: \(label)", false, "ADMITTED") }
        catch { check("refused: \(label)", true, String("\(error)".prefix(150))) }
    }

    // cancellation: cancel a 30 s request mid-run
    let task = Task { try await engine.run(AudioSeparationRequest(audio: longMix, targetDescription: "speech", metaData: ["seed": .int(8)]), package: pid) }
    try await Task.sleep(nanoseconds: 1_500_000_000)
    t0 = Date()
    task.cancel()
    do { _ = try await task.value; check("cancel mid-run", false, "ran to completion") }
    catch { check("cancel mid-run", true, "stopped \(f(Date().timeIntervalSince(t0))) s after cancel (\(String("\(error)".prefix(80))))") }

    // still usable after a cancel
    try await run("after-cancel", AudioSeparationRequest(audio: mix, targetDescription: "speech", metaData: ["seed": .int(7)]),
                  expect: [.target, .residual], length: mixSamples.count, mixRMS: mixRMS)
    await engine.evict(package: pid)
    MLX.Memory.clearCache()
    let p = physGB()
    print("evicted: phys \(f(p.now)) GB (lifetime peak \(f(p.peak)) GB)")
    print("FOOTPRINT resident \(f(resident)) GB, peak activation \(f(afterLong - resident)) GB (30 s envelope), \(f(afterShort - resident)) GB (≤ 10 s)")
    return ok
}
