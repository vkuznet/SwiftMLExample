// load python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

import TensorFlow

/// A batch of examples from the iris dataset.
struct DataBatch {
    /// [batchSize, featureCount] tensor of features.
    let features: Tensor<Float>

    /// [batchSize] tensor of labels.
    let labels: Tensor<Int32>
}

/// Conform `DataBatch` to `Collatable` so that we can load it into a `TrainingEpoch`.
extension DataBatch: Collatable {
    public init<BatchSamples: Collection>(collating samples: BatchSamples)
        where BatchSamples.Element == Self {
        /// `DataBatch`es are collated by stacking their feature and label tensors
        /// along the batch axis to produce a single feature and label tensor
        features = Tensor<Float>(stacking: samples.map{$0.features})
        labels = Tensor<Int32>(stacking: samples.map{$0.labels})
    }
}

/// Initialize an `DataBatch` dataset from a CSV file.
func loadDatasetFromCSV (
        contentsOf: String, hasHeader: Bool, featureColumns: [Int], labelColumns: [Int]) -> [DataBatch] {
    let np = Python.import("numpy")

    let featuresNp = np.loadtxt(
        contentsOf,
        delimiter: ",",
        skiprows: hasHeader ? 1 : 0,
        usecols: featureColumns,
        dtype: Float.numpyScalarTypes.first!)
    guard let featuresTensor = Tensor<Float>(numpy: featuresNp) else {
        // This should never happen, because we construct featuresNp in such a
        // way that it should be convertible to tensor.
        fatalError("np.loadtxt result can't be converted to Tensor")
    }

    let labelsNp = np.loadtxt(
        contentsOf,
        delimiter: ",",
        skiprows: hasHeader ? 1 : 0,
        usecols: labelColumns,
        dtype: Int32.numpyScalarTypes.first!)
    guard let labelsTensor = Tensor<Int32>(numpy: labelsNp) else {
        // This should never happen, because we construct labelsNp in such a
        // way that it should be convertible to tensor.
        fatalError("np.loadtxt result can't be converted to Tensor")
    }

    return zip(featuresTensor.unstacked(), labelsTensor.unstacked()).map{DataBatch(features: $0.0, labels: $0.1)}

}

// TFModel provides TensorFlow model we'll use for training
// the model has three layers
// (4, 10, relu) -> (10, 10, relu) -> (10, 3)
struct TFModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 10, activation: relu)
    var layer3 = Dense<Float>(inputSize: 10, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

// saving layer protocol weights, see
// https://gist.github.com/kongzii/62b9d978a6536bb97095ed3fb74e30fd
// later we should switch to Checkpoints Reader/Writer, see
// https://github.com/tensorflow/swift-models/tree/master/Checkpoints
extension Layer {
    mutating public func loadWeights(numpyFile: String) {
        print("loading weights from: \(numpyFile).npy")
        let np = Python.import("numpy")
        let weights = np.load(numpyFile+".npy", allow_pickle: true)

        for (index, kp) in self.recursivelyAllWritableKeyPaths(to:  Tensor<Float>.self).enumerated() {
            self[keyPath: kp] = Tensor<Float>(numpy: weights[index])!
        }
    }

    public func saveWeights(numpyFile: String) {
        print("saving weights to: \(numpyFile).npy")
        var weights: Array<PythonObject> = []

        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            weights.append(self[keyPath: kp].makeNumpyArray())
        }

        let np = Python.import("numpy")
        np.save(numpyFile, np.array(weights))
    }
}

// helper function to calculate accuracy of our model
func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

// helper function to print our predictions
func printPredictions(classNames: [String], preds: Tensor<Float>) {
    for i in 0..<preds.shape[0] {
        let logits = preds[i]
        let classIdx = logits.argmax().scalar!
        print("Example \(i) prediction: \(classNames[Int(classIdx)]) (\(softmax(logits)))")
    }
}
