using MLDatasets, Flux
using Plots, Images
using Statistics


# load full training set
train_x, train_y = MNIST.traindata(Float32;dir = "./MNIST")

# load full test set
test_x,  test_y  = MNIST.testdata(Float32;dir = "./MNIST")

# Reshape Data in order to flatten each image into a linear array
xtrain = Flux.flatten(train_x)
xtest = Flux.flatten(test_x)

# One-hot-encode the labels
ytrain, ytest = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9)

# Print the dimensions of training feature matrices and training label matrices
println("xtrain dimensions = $(size(xtrain))")
println("ytrain dimensions = $(size(ytrain))")


# Get the dimensions of train_x
(m, n, z) = size(train_x)

# Chain together functions!
model = Flux.Chain(
                    Dense(m*n, 60, Flux.σ),
                    Dense(60, 60, Flux.σ),
                    Dense(60, 10, Flux.σ),
                )

# Define mean squared error loss function
loss(x, y) = Flux.Losses.mse(model(x), y)

# Define the accuracy 
accuracy(x, y) = Statistics.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# ADAM would be the perferred optimizer for serious deep learning
opt = Flux.ADAM()

# Define gradient descent optimizer
# Flux.Descent
#opt = Descent(0.23)

# Format your data
data = [(xtrain, ytrain)]

# Collect weights and bias for your model
parameters = Flux.params(model)

println("Old Loss = $(loss(xtrain, ytrain))")
println("Old Accuracy = $(accuracy(xtrain, ytrain)) \n")

# Train the model over one epoch
Flux.train!(loss, parameters, data, opt)


println("New Loss = $(loss(xtrain, ytrain))")
println("New Accuracy = $(accuracy(xtrain, ytrain))")

println("Old Loss = $(loss(xtrain, ytrain))")
println("Old Accuracy = $(accuracy(xtrain, ytrain)) \n")

(m, n) = size(xtrain)

# Train the model over 100_000 epochs
for epoch in 1:100_000
    # Randomly select a entry of training data 
    i = rand(1:n)
    data = [(xtrain[:, i], ytrain[:, i])]

    # Implement Stochastic Gradient Descent 
    Flux.train!(loss, parameters, data, opt)

    # Print loss function values 
    if epoch % 10_000 == 0
        println("Epoch: $(epoch)")
        @show loss(xtrain, ytrain)
        @show accuracy(xtrain, ytrain)
        println()
    end
end

#Check using test data
@show i = rand(1:1_000)

predict(i) = argmax(model(xtest[:, i])) - 1

digit = predict(i)
println("Predict digit: $(digit)")
println("Actual digit : $(argmax(ytest[:, i]) - 1)")

colorview(Gray, test_x[:,:,i]')


#check using own data drawed via paint/photoshop etc
add_dim(x::Array) = reshape(x, (size(x)...,1))


#rotate resize data so the model can interpret the result
function predict_image(image)
    image = image'
    image = imresize(image,(28,28))
    image = Gray.(image)
    image = convert(Array{Float32}, image)
    image = add_dim(image)
    image = Flux.flatten(image)
    image = image[:,1]
    return argmax(model(image))-1
end

#check every test data
@sync while true
    i=1
    while true

        img =  test_x[:,:,i]'
        save("two.png",img)
        image = load("two.png")
        println("Predicted value:$(predict_image(image))")
        i = i+1
        sleep(1)
    end
end

#check custom data named "uno.png"
image = load("uno.png")
println("Predicted value:$(predict_image(image))")
imresize(image,(28,28))