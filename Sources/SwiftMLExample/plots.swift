// load python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

// create plot of our loss function
func plot(acc: [Float], loss: [Float], fname: String) {
    // add matplotlib handler
    let matplotlib = Python.import("matplotlib")
    matplotlib.use("Agg")
    let plt = Python.import("matplotlib.pyplot")

    plt.figure(figsize: [12, 8])

    let accuracyAxes = plt.subplot(2, 1, 1)
    accuracyAxes.set_ylabel("Accuracy")
    accuracyAxes.plot(acc)

    let lossAxes = plt.subplot(2, 1, 2)
    lossAxes.set_ylabel("Loss")
    lossAxes.set_xlabel("Epoch")
    lossAxes.plot(loss)
    //plt.show()
    plt.savefig(fname)
}
