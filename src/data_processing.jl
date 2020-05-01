using CSV, DataFrames
using FileIO: load
using MLDataUtils: stratifiedobs
using ImageTransforms
using Images
using Random: shuffle!, MersenneTwister

function split_data()
    df = DataFrame(CSV.File("Image_Paths.csv"))
    encoding = Dict(
        "Cloud"=>0, "Dust"=>1, "Haze"=>2, "Land"=>3, "Seaside"=>4, "Smoke"=>5
    )
    y = replace(df.image_type, encoding...)
    weights = class_weights(y)
    X = df.image_path
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratifiedobs((X, y), p=(0.64, 0.16), rng=MersenneTwister(42))
    X_train, y_train, X_val, y_val, X_test, y_test, weights
end

function load_images(paths)
    imgs = []
    for path in paths
        push!(imgs, float32.(load(path)))
    end
    imgs
end

function class_weights(y::AbstractVector, classes=1:length(unique(y)))
    counts = map(class -> count(x -> x == class, y), classes)
    replace(Float32.(length(y) / length(classes) ./ counts), Inf=>0)
end

struct Dataset
    data
    transforms
    batchsize
    shuffle
    n_obs
    indices
end

function Dataset(data...; transforms=identity, batchsize=1, shuffle=true)
    n_obs = size(data[1])[end]
    for i in 2:length(data)
        if size(data[i])[end] != n_obs
            throw(DimensionMismatch("all data members must have same number of observations. $i has different number of observations."))
        end
    end
    Dataset(data, transforms, batchsize, shuffle, n_obs, collect(1:n_obs))
end

function Base.getindex(dataset::Dataset, idx)
    checkindex(Bool, 1:dataset.n_obs, idx) || throw(BoundsError(dataset.data[1], idx))
    dataset.transforms(Tuple(x[(Colon() for _ in 2:ndims(x))..., dataset.indices[idx]] for x in dataset.data))
end

function Base.length(dataset::T) where {T<:Dataset}
    dataset.n_obs
end

function Base.firstindex(dataset::T) where {T<:Dataset}
    1
end

function Base.lastindex(dataset::T) where {T<:Dataset}
    dataset.n_obs
end

function Base.iterate(dataset::T, idx=1) where {T<:Dataset}
    idx > dataset.n_obs && return nothing
    if idx==1 && dataset.shuffle
        shuffle!(dataset.indices)
    end
    next_idx = min(idx + dataset.batchsize, dataset.n_obs)
    dataset[idx:next_idx - 1], next_idx
end

function train_transforms()
    crop((img, y)) = (ImageTransforms.random_crop(img, (224, 224)), y)
    hflip((img, y)) = (ImageTransforms.random_horizontal_flip(img), y)
    vflip((img, y)) = (ImageTransforms.random_vertical_flip(img), y)
    normalise((img, y)) = (ImageTransforms.Normalizer(0.5, 0.5)(img), y)
    ((img, y)) -> crop |> hflip |> vflip |> ImageTransforms.img2array |> normalise
end

function test_transforms()
    normalise((img, y)) = (ImageTransforms.Normalizer(0.5, 0.5)(img), y)
    normalise
end
