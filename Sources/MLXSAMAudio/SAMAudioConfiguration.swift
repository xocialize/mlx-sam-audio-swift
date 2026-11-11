// SAMAudioConfiguration.swift — init-time configuration for `SAMAudioPackage` (C9). Per-request description / spans / seed ride
// the canonical `AudioSeparationRequest`.
//
// ONE weight set: the converter's bundle (`Tools/convert`) — `model.safetensors` (fp16 DiT), `text_encoder.safetensors`
// (fp32 T5-base encoder), `codec.safetensors` (fp32 DAC-VAE with the watermark branch), `config.json`, the t5-base tokenizer
// and the SAM `LICENSE` — published as one repository and pinned to its verified commit. `bundleDirectory` is the dev escape
// hatch.

import Foundation
import MLXToolKit

public struct SAMAudioConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The DiT's precision. Only `.fp16` is offered (the published tier; the T5 encoder and the codec are fp32 inside it).
    public var quant: Quant
    /// Revision of the weights repo; nil = the published, hash-verified commit (`revisionPin`).
    public var revision: String?
    /// Explicit bundle directory (the converter's layout) — never touches the network.
    public var bundleDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// MLX buffer-cache limit while the model is resident (the fleet's bounded-cache rule; the footprints were measured at 1 GiB).
    public var cacheLimitBytes: UInt64

    public init(quant: Quant = .fp16, revision: String? = nil, bundleDirectory: URL? = nil, modelsRootDirectory: URL? = nil,
                cacheLimitBytes: UInt64 = 1 << 30) {
        self.quant = quant; self.revision = revision; self.bundleDirectory = bundleDirectory
        self.modelsRootDirectory = modelsRootDirectory; self.cacheLimitBytes = cacheLimitBytes
    }

    /// The published bundle (2026-11-11): hash-equal to the gated local conversion, verified from a fresh download.
    public static let repo = "mlx-community/sam-audio-base-swift-fp16"
    public static let revisionPin = "2cf624a4e250db577661572f07f56b758f46dffd"
    /// The upstream checkpoint the converter read (gated on Hugging Face; the oracle ran on it).
    public static let upstreamRepo = "facebook/sam-audio-base"
    public static let upstreamRevision = "81f64008f9f957c2b57a45923fb5299c66e9d186"

    public var effectiveRevision: String { revision ?? Self.revisionPin }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case quant, revision, cacheLimitBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .fp16
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        cacheLimitBytes = try c.decodeIfPresent(UInt64.self, forKey: .cacheLimitBytes) ?? (1 << 30)
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension SAMAudioConfiguration: WeightSourcing {
    /// The files `SAMAudioPipeline` opens, plus the licence that must travel with the weights (SAM License §1b-i).
    public static let files = ["*.json", "*.safetensors", "spiece.model", "LICENSE"]
    /// Representative file for the missing-probe (the last component `load()` opens).
    static let probe = "codec.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "bundle", repo: Self.repo, revision: effectiveRevision, matching: Self.files)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let bundleDirectory {
            return FileManager.default.fileExists(atPath: bundleDirectory.appendingPathComponent(Self.probe).path) ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The bundle directory `load()` opens: the explicit one, else the repo's snapshot (hub-client layout) or flat directory
    /// (engine-executed layout) under the store. Nil when it does not resolve.
    public func resolvedBundle(storeRoot: URL?) -> URL? {
        if let bundleDirectory { return bundleDirectory }
        let store = ModelStore(root: storeRoot)
        func has(_ dir: URL) -> Bool { FileManager.default.fileExists(atPath: dir.appendingPathComponent(Self.probe).path) }
        if let snapshot = store.snapshotDirectory(for: Self.repo, revision: effectiveRevision), has(snapshot) { return snapshot }
        if let flat = store.directory(for: Self.repo), has(flat) { return flat }
        return nil
    }
}

// MARK: - Cold-start prewarm

extension SAMAudioConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { resolvedBundle(storeRoot: modelsRootDirectory).map { [$0] } ?? [] }
}
