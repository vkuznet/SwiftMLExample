## SwiftMLExample
Repository with end-to-end ML example based on
[TF Swift tutorial](https://www.tensorflow.org/swift/tutorials/model_training_walkthrough).
We extend the tutorial to provide example how to wrap up everything
into single package, organize codebase into separate modules,
add external arguments, etc.

### Build notes
Swift provides [package manager](https://swift.org/getting-started/#using-the-package-manager)
which helps to setup initial project area. To do that just create a new
directory and run within it the following command:
```
# initalize the new package (it is already done here)
swift package init --type executable
```
This will initialize the project and create Package.swift file which
you can later customize to include your set of dependencies, etc.
This is already done for this package.

To maintain the codebase you need to use the following commands (within
this project area):
```
# build/compile our codebase (the entire build will be located in .build area)
swift build

# run our executable
swift run swift-ml --help

# run our training for 300 epochs using 64 batch size and save model to model.tf file
swift run swift-ml train -e 300 --batch-size 64 --model-filename model.tf

# run test ML action, i.e. load the model and make predictions
swift run swift-ml test --model-filename model.tf

# run MNIST training
swift run swift-ml mnist

# clean-up our build
swift package clean

# create full release
swift build --configuration release

# grab new executable from release area and put it into provide path
cp .build/release/swift-ml /path
```

Further customization can be done using swift
[Mint](https://github.com/yonaskolb/Mint) package.
