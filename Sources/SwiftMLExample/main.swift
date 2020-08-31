import ArgumentParser

// for help, please see https://github.com/apple/swift-argument-parser
struct SwiftML: ParsableCommand {
    @Option(name: .shortAndLong, help: "The number of epochs for train (default 500)")
    var epochs: Int?
    @Option(name: .shortAndLong, help: "The batch size (default 32)")
    var batchSize: Int?
    @Option(name: .shortAndLong, help: "Model file name")
    var modelFilename: String?
    @Argument(help: "Perform ML action (train|test)")
    var action: String
    mutating func run() throws {
        if action == "train" {
            train(epochs ?? 500, batchSize ?? 32, modelFilename ?? "")
        } else if action == "test" {
            test(modelFilename ?? "")
        } else if action == "mnist" {
            trainMNIST()
        } else {
            print("unsupported action \(action)")
        }
    }
}

SwiftML.main()
