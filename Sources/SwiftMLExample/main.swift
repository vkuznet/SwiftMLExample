import Foundation
import TensorFlow
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// import Python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif
print(Python.version)

// list current content of our directory
if #available(macOS 10.13, *) {
    print("/bin/ls".shell("-lh"))
}

// download train and test dataset
let url = "http://download.tensorflow.org/data"
let trainDataFilename = "iris_training.csv"
let testDataFilename = "iris_test.csv"
downloadData(from: url+"/"+testDataFilename, to: testDataFilename)
downloadData(from: url+"/"+trainDataFilename, to: trainDataFilename)

// inspect the data we just donwloaded
inspectData(fname: trainDataFilename)
inspectData(fname: testDataFilename)

let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
let labelName = "species"
let columnNames = featureNames + [labelName]

print("Features: \(featureNames)")
print("Label: \(labelName)")

let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]

let batchSize = 32

let trainingDataset: [DataBatch] = loadDatasetFromCSV(contentsOf: trainDataFilename,
                                                  hasHeader: true,
                                                  featureColumns: [0, 1, 2, 3],
                                                  labelColumns: [4])

let trainingEpochs: TrainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchSize)

let firstTrainEpoch = trainingEpochs.next()!
let firstTrainBatch = firstTrainEpoch.first!.collated
let firstTrainFeatures = firstTrainBatch.features
let firstTrainLabels = firstTrainBatch.labels

//print("First batch of features: \(firstTrainFeatures)")
//print("firstTrainFeatures.shape: \(firstTrainFeatures.shape)")
//print("First batch of labels: \(firstTrainLabels)")
//print("firstTrainLabels.shape: \(firstTrainLabels.shape)")

// create TF model
var model = TFModel()

/*
// Apply the model to a batch of features.
let firstTrainPredictions = model(firstTrainFeatures)
// our prediction
print(firstTrainPredictions[0..<5])
// pass through softmax
print(softmax(firstTrainPredictions[0..<5]))
print("Prediction: \(firstTrainPredictions.argmax(squeezingAxis: 1))")
print("    Labels: \(firstTrainLabels)")
*/

// choose loss function
let untrainedLogits = model(firstTrainFeatures)
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: firstTrainLabels)
print("Loss test: \(untrainedLoss)")

// choose optimizer
let optimizer = SGD(for: model, learningRate: 0.01)

let (loss, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
print("Current loss: \(loss)")

optimizer.update(&model, along: grads)

let logitsAfterOneStep = model(firstTrainFeatures)
let lossAfterOneStep = softmaxCrossEntropy(logits: logitsAfterOneStep, labels: firstTrainLabels)
print("Next loss: \(lossAfterOneStep)")

// training loop
let epochCount = 500
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

for (epochIndex, epoch) in trainingEpochs.prefix(epochCount).enumerated() {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batchSamples in epoch {
        let batch = batchSamples.collated
        let (loss, grad) = valueWithGradient(at: model) { (model: TFModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.labels)
        }
        optimizer.update(&model, along: grad)

        let logits = model(batch.features)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)
    if epochIndex % 50 == 0 {
        print("Epoch \(epochIndex): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}

// create accuracy and loss plot
plot(acc: trainAccuracyResults, loss: trainLossResults, fname: "loss.pdf")

// proceed with test dataset
let testDataset = loadDatasetFromCSV(
    contentsOf: testDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]).inBatches(of: batchSize)

// evaluate model on test dataset
// NOTE: Only a single batch will run in the loop since the batchSize we're using is larger than the test set size
for batchSamples in testDataset {
    let batch = batchSamples.collated
    let logits = model(batch.features)
    let predictions = logits.argmax(squeezingAxis: 1)
    print("Test batch accuracy: \(accuracy(predictions: predictions, truths: batch.labels))")
}

/*
// first batch accuracy
let firstTestBatch = testDataset.first!.collated
let firstTestBatchLogits = model(firstTestBatch.features)
let firstTestBatchPredictions = firstTestBatchLogits.argmax(squeezingAxis: 1)
print("First batch accuracy:")
print(firstTestBatchPredictions)
print(firstTestBatch.labels)
*/

// make predictions
let unlabeledDataset: Tensor<Float> =
    [[5.1, 3.3, 1.7, 0.5],
     [5.9, 3.0, 4.2, 1.5],
     [6.9, 3.1, 5.4, 2.1]]
printPredictions(preds: model(unlabeledDataset))

// save our model to external file which we may use later
let modelFileName = "model.tf"
model.saveWeights(numpyFile: modelFileName)

// let's load model back and re-do predictions
var newModel = TFModel()
newModel.loadWeights(numpyFile: modelFileName)
printPredictions(preds: newModel(unlabeledDataset))
