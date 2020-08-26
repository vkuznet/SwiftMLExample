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
func loadIrisDatasetFromCSV (
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

// our IrisModel
struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

