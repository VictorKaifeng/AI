# We want to use AI to find a line that breaks R^2 with y=-xlim
using DelimitedFiles
using Plots
using LinearAlgebra

function activation(s)
    if (s<0)
        return -1
    end
    if (s>0)
        return 1
    end
end

function Δ(w, des_x, x)
    return des_x .*x
end

function Δθ(θ, des_x, x)
    return des_x
end

function correction(w,θ, des_x, x)
    w = w .+ Δ(w, des_x, x)
    θ = θ .+ Δθ(θ, des_x, x)
    return (w,θ)
end
function perceptron(w,θ, des_x, x)
    s = dot(w,x)+θ
    y= activation(s)
    if ( y == des_x)
        return (w,θ)
    end
    (w,θ) =  correction(w, θ, des_x, x)
    perceptron(w, θ, des_x, x)
end

function AI_main()
    dat = readdlm("pesos_julia.txt", Float64)
    w = zeros(2)
    w[1] = dat[1]
    w[2] = dat[2]
    x = zeros(2)
    θ = zeros(1)
    θ = dat[3]
    x = (0.5 , .5)
    des_x = 1
    (w,θ) = perceptron(w,θ,des_x,x)
    i = 0
    for i in 1:2000
        x = rand(Float64,2)
        x[1] = rand(-10:10)
        x[2] = rand(-10:10)
        des_x = 1
        if (x[2]+ x[1] >= 1)
                 des_x = -1
        end
        (w,θ) = perceptron(w,θ,des_x,x)
    end
    y = zeros(2)
    #println("Escoge dos numeros")
    #num  = readline() 
    num= "3"
    y[1] = parse(Float64, num)
    #num  = readline() 
    y[2] = parse(Float64, num)
    y = (0.3,0.5)
    open("pesos_julia.txt","w") do io
        println(io,w[1])
        println(io,w[2])
        println(io,θ)
     end
     x = range(-10,10)
    f(x)= -w[1]/w[2] .* x .-θ/w[2]
    plt = plot(x,f(x))
    plt = xlims!(-10,10)
    plt = ylims!(-10,10)
    return (w,θ, plt)
end



function main()

    (w, θ, plt) = AI_main()
    y = (1,-0.03)
    if (dot(w,y)+θ>=0)
        #println("Y mas grande que X") 
    else
        #println("X mas grande que Y") 
    end
    return plt
end

@gif for i ∈ 1:10
    main()
end every 1