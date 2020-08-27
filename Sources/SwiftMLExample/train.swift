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

// helper function to train ML model
func train(_ nEpochs: Int = 10, _ batchSize: Int = 32, _ modelFilename: String) {
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
    // let columnNames = featureNames + [labelName]

    print("Features: \(featureNames)")
    print("Label: \(labelName)")

    let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    let trainingDataset: [DataBatch] = loadDatasetFromCSV(contentsOf: trainDataFilename,
                                                      hasHeader: true,
                                                      featureColumns: [0, 1, 2, 3],
                                                      labelColumns: [4])

    let trainingEpochs: TrainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchSize)

    let firstTrainEpoch = trainingEpochs.next()!
    let firstTrainBatch = firstTrainEpoch.first!.collated
    let firstTrainFeatures = firstTrainBatch.features
    let firstTrainLabels = firstTrainBatch.labels

    // create TF model
    var model = TFModel()

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
    var trainAccuracyResults: [Float] = []
    var trainLossResults: [Float] = []

    for (epochIndex, epoch) in trainingEpochs.prefix(nEpochs).enumerated() {
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

    // make predictions
    printPredictions(classNames: classNames, preds: model(unlabeledData()))
    // save our model to external file which we may use later
    if modelFilename != "" {
        model.saveWeights(numpyFile: modelFilename)
    }
}

// helper function to provide unlabeled data for testing ML model
func unlabeledData() -> Tensor<Float> {
    let dataset: Tensor<Float> =
        [[5.1, 3.3, 1.7, 0.5],
         [5.9, 3.0, 4.2, 1.5],
         [6.9, 3.1, 5.4, 2.1]]
    return dataset
}

// helper function to test fiven model file name with unlabeled data
func test(_ modelFilename: String) {
    let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    if modelFilename != "" {
        // let's load model our model
        var model = TFModel()
        model.loadWeights(numpyFile: modelFilename)
        printPredictions(classNames: classNames, preds: model(unlabeledData()))
    } else {
        print("there is no model file name provided, skip the test")
    }
}
