using Flux, Plots, CSV, DataFrames
#For log-time series study
function point_to_log(point)
    point2 = copy(point)
    p = length(point)
    for i in p:-1:2
        point2[i] = log.(point2[i]./point2[i-1])
    end
    point2[1] = point2[1].*0
    return point2
end
#undo point_to_log
function point_to_exp(point,inicial)
    point2 = copy(point)
    p = length(point)
    point2[1] =  inicial.*1.0
    for i in 2:p
        point2[i] = exp.(point2[i])*point2[i-1]
    end
    return point2
end
#good for constant patterns, modify "Ibex.csv" for desired data set
data = CSV.read("^IBEX.csv", DataFrame)
#Only using last 56 days
price = Float32.(data.Close)[200:256]
#Convert data to log-timeseries
price2 = point_to_log(price)
#Data gathering
data_x =  [x for x in 1:1:51 ]
data_y = [ y for y in price2]
#Model creation
time_window = 20
future = 10
model = Chain(
            LSTM(time_window =>23), 
            LSTM(23=>1)
        )
# Model Parameters
parameters = Flux.params(model)

optim = ADAM(0.001)

loss(m,x,y) = (m(x) .- y) .^2

#keep track of best
epoch_loss = 0
epoch_best = deepcopy(model)
loss_epoch = NTuple{1,Float32}[]

#training
function train_y(n)
    #reset recursive layer inside model
    Flux.reset!(model)

    #actual training
    for epoch in 1:n
    local i = rand(1+time_window:size(data_y)[1])
    grad = Flux.gradient(parameters) do 
        epoch_loss =  sum(loss(model,data_y[i-time_window:i-1],data_y[i]))
        return epoch_loss
    end

    # loss recording
    if (epoch%500 == 0)
        local epoch_loss =  sum(loss(model,data_y[i-time_window:i-1],data_y[i]))
        local point = (epoch_loss,)
        push!(loss_epoch,point)
    end

    #keep the best
    if (sum(loss(model,data_y[i-time_window:i-1],data_y[i]))<= sum(loss(epoch_best,data_y[i-time_window:i-1],data_y[i])))
        global epoch_best = deepcopy(model)
    end

    #update 
    Flux.update!(optim, parameters, grad)
    end
end

# Plotting the results as an animation
#=
anim = @animate for i in 1:10000
train(10)
a = plot(x -> f(x), -5, 5, legend=true, label = "f")
#a = plot!(x -> model([x]), -5:0.1f0:5, label = "model")
a = plot!(x -> epoch_best([x]), -5:0.1f0:5, label = "best model")
end

gif(anim, "15fpstes.gif",fps = 15)=#

#Plotting the results directly
    train_y(10_000)
    # a will be log plot 
    a = plot(price2 , label ="log-price")

    a = plot!(size=(1920,1800))
    
    #Timeseries modeling requieres a starting set
    plot_model = zeros(size(price)[1]+future)
    for i in 1:time_window
        plot_model[i] =  price2[i]
    end
    #auxiliar vector
    vector  = zeros(time_window)
    #compute model prediction
    for i in time_window+1:size(data_x)[1]+future
        for j in  i-time_window:i-1
            vector[time_window-(i-j)+1] = plot_model[j][1] 
        end
        y_value = model(vector)[1]
        plot_model[i] =   y_value
    end

    #add model to plot a
    a = plot!(plot_model,  label = "best model")

    #modify data to plot non log model
    plot_model = point_to_exp(plot_model,price[1])
    # set b as non log model
    b = plot(price, label = "real price", title = "TW: 20 days, Ibex-35: 11-10-2023 to 13-01-2024 ")
    b = plot!(plot_model,  label = "best model")
    #plot a and b
    plot(a,b)



    #=
    open("indices_log.txt","w") do io
        for i in 1:size(plot_model)[1]
            println(io,plot_model[i][2])
        end
     end
=#


     
#end
#gif(anim, "15fpstes.gif",fps = 15)

######################################
#=
time_window = 50
model2 = Chain(
            RNN(time_window =>23), 
            Dense(23=>1, bias=false),
            only
        )
global epoch_best2 = deepcopy(model2)
function train_x(n,model)
    Flux.reset!(model)
    for epoch in 1:n
        #actual training
    local i = rand(1+time_window:size(data_y)[1])
    grad = Flux.gradient(parameters) do 
        epoch_loss =  sum(loss(model2,data_x[i-time_window:i-1],data_y[i]))
        return epoch_loss
    end
    # loss recording
    if (epoch%500 == 0)
        local epoch_loss =  sum(loss(model2,data_x[i-time_window:i-1],data_y[i]))
        local point = (epoch_loss,)
        push!(loss_epoch,point)
    end
    #keep the best
    if (sum(loss(model2,data_x[i-time_window:i-1],data_y[i]))<= sum(loss(epoch_best2,data_x[i-time_window:i-1],data_y[i])))
        global epoch_best2 = deepcopy(model2)
    end
    #update 
    Flux.update!(optim, parameters, grad)
    end
end


train_x(100,model2)
c = plot(x -> f(x), 0, 4, legend=true, label = "f", title = "X")
plot_model = NTuple{2,Float64}[]
#set plot model to start
for i in 1:time_window
    push!( plot_model, (data_x[i],data_y[i]))
end
vector  = zeros(time_window)
#compute model prediction
for i in time_window+1:size(data_x)[1]+200
    for j in  i-time_window:i-1
        vector[time_window-(i-j)+1] = plot_model[j][2]
    end
push!(plot_model , (data_x[time_window+1]+i*0.001,epoch_best2(vector)[1]))
end
plot_model
c = plot!(plot_model,  label = "best model")
plot(a,c)=#

