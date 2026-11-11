// swift-tools-version: 6.2
// mlx-sam-audio-swift — Meta's SAM-Audio (text- and span-prompted source separation) on Swift-MLX.
// SAMAudioCore is the model (T5-base text encoder, DiT, DAC-VAE codec with its watermark, flow solver); MLXSAMAudio is the
// MLXEngine package on `audioSeparation` (contract ≥ 1.62.0: a described target + time spans); samaudio-gates is the parity
// and validation CLI.
import PackageDescription

let package = Package(
    name: "mlx-sam-audio-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "SAMAudioCore", targets: ["SAMAudioCore"]),
        .library(name: "MLXSAMAudio", targets: ["MLXSAMAudio"]),
        .executable(name: "samaudio-gates", targets: ["samaudio-gates"]),
    ],
    dependencies: [
        // The fleet's fork, as mlx-engine-swift ≥ 0.74.1 uses it (AB-D-0208): two URLs for one package identity is an error.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.32.4"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.77.0"),
    ],
    targets: [
        .target(
            name: "SAMAudioCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/SAMAudioCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXSAMAudio",
            dependencies: [
                "SAMAudioCore",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXSAMAudio",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "samaudio-gates",
            dependencies: [
                "SAMAudioCore", "MLXSAMAudio",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/samaudio-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SAMAudioCoreTests",
            dependencies: ["SAMAudioCore"],
            path: "Tests/SAMAudioCoreTests"
        ),
        .testTarget(
            name: "MLXSAMAudioTests",
            dependencies: [
                "SAMAudioCore", "MLXSAMAudio",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXSAMAudioTests"
        ),
    ]
)
