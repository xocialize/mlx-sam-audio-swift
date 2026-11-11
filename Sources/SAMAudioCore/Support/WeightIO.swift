// WeightIO.swift — safetensors in, key contract enforced (the fleet idiom, from mlx-heartmula-swift): load on the CPU stream,
// cast floating tensors per the dtype policy, compare the module's flattened parameter paths with the on-disk keys — 0 missing,
// 0 unused, or throw — then `update(verify:)`. A partial load is the silent-failure class.
//
// Bundle layout = Tools/convert: `model.safetensors` (DiT, projections, anchors, the align terms; torch key names, convs
// (O, K, I)), `text_encoder.safetensors` (T5-base encoder, compact keys), `codec.safetensors`, `config.json`.
import Foundation
import MLX
import MLXNN

public enum SAMAudioError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        }
    }
}

public enum WeightIO {
    /// Loads every array of one safetensors file on the CPU stream (a lazy GPU-stream load of a multi-GB file trips the Metal
    /// watchdog), casting floating tensors to `dtype` when given.
    public static func load(_ url: URL, dtype: DType? = nil) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw SAMAudioError.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                if floating, let d = dtype, v.dtype != d { out[k] = v.asType(d) } else { out[k] = v }
            }
            eval(Array(out.values))
            return out
        }
    }

    /// The arrays under `prefix`, with the prefix removed.
    public static func scoped(_ arrays: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (k, v) in arrays where k.hasPrefix(prefix) { out[String(k.dropFirst(prefix.count))] = v }
        return out
    }

    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String) throws {
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw SAMAudioError.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
    }
}

/// `config.json` as the converter writes it (the upstream config's text path).
public struct SAMAudioConfig: Decodable, Sendable {
    public struct Transformer: Decodable, Sendable {
        public let dim: Int
        public let n_heads: Int
        public let n_layers: Int
        public let norm_eps: Float
        public let ffn_exp: Int
        public let ffn_dim_multiplier: Int
        public let multiple_of: Int
        public let max_positions: Int
        public let frequency_embedding_dim: Int
        public let out_channels: Int
    }
    public struct TextEncoder: Decodable, Sendable {
        public let dim: Int
        public let layers: Int
        public let heads: Int
        public let d_kv: Int
        public let d_ff: Int
        public let vocab: Int
        public let relative_buckets: Int
        public let relative_max_distance: Int
        public let max_length: Int
    }
    public let in_channels: Int
    public let transformer: Transformer
    public let text_encoder: TextEncoder
    public let num_anchors: Int
    public let anchor_embedding_dim: Int

    public static func load(_ url: URL) throws -> SAMAudioConfig {
        guard FileManager.default.fileExists(atPath: url.path) else { throw SAMAudioError.missingFile(url.path) }
        return try JSONDecoder().decode(SAMAudioConfig.self, from: Data(contentsOf: url))
    }
}
