// swift-tools-version: 6.2
// mlx-sam-audio-swift — Meta's SAM-Audio (text- and span-prompted source separation) on Swift-MLX.
// SAMAudioCore is the model (T5-base text encoder, DiT, DAC-VAE codec, flow solver); samaudio-gates is the parity CLI.
// The engine wrapper (MLXSAMAudio) lands at V3.
import PackageDescription

let package = Package(
    name: "mlx-sam-audio-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "SAMAudioCore", targets: ["SAMAudioCore"]),
        .executable(name: "samaudio-gates", targets: ["samaudio-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
        .executableTarget(
            name: "samaudio-gates",
            dependencies: [
                "SAMAudioCore",
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
    ]
)
