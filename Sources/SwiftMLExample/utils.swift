import Foundation

// helper function to run shell command of the specified string
// "/bin/ls".shell("-lh")
@available(macOS 10.13, *)
public extension String {
    @discardableResult
    func shell(_ args: String...) -> String
    {
        let (task,pipe) = (Process(),Pipe())
        task.executableURL = URL(fileURLWithPath: self)
        (task.arguments,task.standardOutput) = (args,pipe)
        do    { try task.run() }
        catch { print("Unexpected error: \(error).") }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: String.Encoding.utf8) ?? ""
    }
}

// helper function to download files
func download(from sourceString: String, to destinationString: String) {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: destinationString) {
        print("File \(destinationString) exists")
        return
    }

    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
}

