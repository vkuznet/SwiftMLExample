// load python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

import TensorFlow
import Foundation
import Just
import Path

public func downloadFile(_ url: String, dest: String? = nil, force: Bool = false) {
    let dest_name = dest ?? (Path.cwd/url.split(separator: "/").last!).string
    let url_dest = URL(fileURLWithPath: (dest ?? (Path.cwd/url.split(separator: "/").last!).string))
    if !force && Path(dest_name)!.exists { return }

    print("Downloading \(url)...")

    if let cts = Just.get(url).content {
        do    {try cts.write(to: URL(fileURLWithPath:dest_name))}
        catch {print("Can't write to \(url_dest).\n\(error)")}
    } else {
        print("Can't reach \(url)")
    }
}

// from fast.ai course
protocol ConvertibleFromByte: TensorFlowScalar {
    init(_ d:UInt8)
}
extension Float : ConvertibleFromByte {}
extension Int32 : ConvertibleFromByte {}
extension Data {
    func asTensor<T:ConvertibleFromByte>() -> Tensor<T> {
        return Tensor(map(T.init))
    }
}
func loadMNIST<T: ConvertibleFromByte>
            (training: Bool, labels: Bool, path: Path, flat: Bool) -> Tensor<T> {
    let split = training ? "train" : "t10k"
    let kind = labels ? "labels" : "images"
    let batch = training ? 60000 : 10000
    let shape: TensorShape = labels ? [batch] : (flat ? [batch, 784] : [batch, 28, 28])
    let dropK = labels ? 8 : 16
    let baseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    let fname = split + "-" + kind + "-idx\(labels ? 1 : 3)-ubyte"
    let file = path/fname
    if !file.exists {
        downloadFile("\(baseUrl)\(fname).gz", dest:(path/"\(fname).gz").string)
        "/bin/gunzip".shell("-fq", (path/"\(fname).gz").string)
    }
    let data = try! Data(contentsOf: URL(fileURLWithPath: file.string)).dropFirst(dropK)
    if labels { return data.asTensor() }
    else      { return data.asTensor().reshaped(to: shape)}
}

public func loadMNIST(path:Path, flat:Bool = false)
        -> (Tensor<Float>, Tensor<Int32>, Tensor<Float>, Tensor<Int32>) {
    try! path.mkdir(.p)
    return (
        loadMNIST(training: true,  labels: false, path: path, flat: flat) / 255.0,
        loadMNIST(training: true,  labels: true,  path: path, flat: flat),
        loadMNIST(training: false, labels: false, path: path, flat: flat) / 255.0,
        loadMNIST(training: false, labels: true,  path: path, flat: flat)
    )
}
