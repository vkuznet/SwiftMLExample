// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "SwiftMLExample",
    platforms: [
        .macOS(.v10_15),
    ],
    products: [
        .executable(name: "swift-ml", targets: ["SwiftMLExample"]),
    ],
    dependencies: [
        //.package(url: "https://github.com/apple/swift-argument-parser", from: "0.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "0.2.0"),
        .package(url: "https://github.com/mxcl/Path.swift", from: "1.2.0"),
        .package(url: "https://github.com/JustHTTP/Just", from: "0.7.2"),
        //.package(url: "https://github.com/tensorflow/swift-models", .revision("3acbabd23a8d9b09aaf6d3c38391d0cbed7ce7b9")),
        // example of using local package, for that we need its location
        // and git revision hash string
        .package(url: "../tmp/swift-models", .revision("afc34e82633896d0e482243db732e1e79be6b520")),
    ],
    targets: [
        .target(
            name: "SwiftMLExample",
            dependencies: [
                "Just",
                .product(name: "Datasets", package: "swift-models"),
                .product(name: "TrainingLoop", package: "swift-models"),
                .product(name: "Path", package: "Path.swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]),
        .testTarget(
            name: "SwiftMLExampleTests",
            dependencies: ["SwiftMLExample"]),
    ]
)
