using DeepLearningModels
using BSON
using Flux, CuArrays
using Metrics
include("data_processing.jl")

function test_model(model, test_loader; use_gpu=true)
    if use_gpu
        to_gpu = gpu
        CuArrays.allowscalar(false)
    else
        to_gpu = identity
    end
    model = to_gpu(model)
    testmode!(model)
    ŷ = Float32[]
    y_true = Float32[]
    for (X_test, y_test) in test_loader
        y = cpu(softmax(model(to_gpu(X_test)); dims=1))
        y = mapslices(argmax, y; dims=1)
        y = reshape(y, :)
        append!(y_true, y_test)
        append!(ŷ, y)
    end
    kappa = cohen_kappa(y_true, ŷ, [1:6;])
    f1 = f1_score(y_true, ŷ, [1:6;])
    println("Cohen's Kappa: $kappa")
    println("F₁ score: $f1")
    model = cpu(model)
end

paths_train, y_train, paths_valid, y_valid, paths_test, y_test, weights = split_data()
X_test = load_images(paths_test)

map!(img -> imresize(img, 230, 230), X_test, X_test)
test_dataset = Dataset(X_test, y_test; transforms=test_transforms(), batchsize=32)

model = BSON.load("model.bson")[:model]
testmode!(model)

test_model(model, test_dataset)
