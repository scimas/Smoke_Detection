using DeepLearningModels
using Flux, CuArrays
using Flux.Optimise: update!
using BSON
using Dates: now

include("data_processing.jl")

function train!(model, loss, optimizer, train_loader, valid_loader; class_weights=1, epochs=100, use_gpu=true)
    if use_gpu
        to_gpu = gpu
        CuArrays.allowscalar(false)
    else
        to_gpu = identity
    end
    model = to_gpu(model)
    class_weights = to_gpu(class_weights)
    min_val_loss = 1f10
    println("Starting training loop.")
    for epoch in 1:epochs
        tick = now()
        trainmode!(model)
        weights = Flux.params(model)
        train_loss = 0f0
        for (x, y) in train_loader
            grads = gradient(weights) do
                minibatch_loss = loss(model(to_gpu(x)), to_gpu(y), class_weights)
                train_loss += cpu(minibatch_loss) * size(y, 2)
                minibatch_loss
            end
            update!(optimizer, weights, grads)
        end
        testmode!(model)
        valid_loss = 0f0
        for (x, y) in valid_loader
            valid_loss += cpu(loss(model(to_gpu(x)), to_gpu(y), class_weights))
        end
        tock = now()
        println("Epoch: $(epoch) | Training loss: $(train_loss / train_loader.n_obs) | Validation loss: $(valid_loss / valid_loader.n_obs) | Time: $(tock - tick)")
        if valid_loss < min_val_loss
            min_val_loss = valid_loss
            model = cpu(model)
            bson("model.bson", Dict(:model => model))
            model = to_gpu(model)
        end
    end
    model = cpu(model)
    println("Training complete.")
end

paths_train, y_train, paths_valid, y_valid, paths_test, y_test, weights = split_data()
X_train = load_images(paths_train)
X_valid = load_images(paths_valid)

map!(img -> imresize(img, 230, 230), X_train, X_train)
map!(img -> imresize(img, 224, 224), X_valid, X_valid)
X_train = reduce((x, y) -> cat(x, y; dims=3), X_train)
X_valid = reduce((x, y) -> cat(x, y; dims=3), X_valid)

train_dataset = Dataset(X_train, y_train; transforms=train_transforms(), batchsize=16)
valid_dataset = Dataset(X_valid, y_valid; transforms=test_transforms(), batchsize=32)

model = SatelliteNet("s")
optimizer = Flux.Optimise.ADAMW(0.0001, (0.9, 0.999), 1.0)
loss(ŷ, y, w) = Flux.logitcrossentropy(ŷ, y, w)

train!(model, loss, optimizer, train_dataset, valid_dataset; class_weights=weights)
