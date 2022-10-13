# This program assumes that Eve attacks only one side

using LinearAlgebra, Optim, Plots

function xlog2x(x)
    if x<0 || x>1
        NaN
    elseif x==0 || x==1
        0
    else
        -x*log2(x)
    end
end

function hshannon(prob)
    sum(xlog2x.(prob))
end

function hbin(x)
    hshannon((x,1-x))
end

function hcond(pab,pb)
     hshannon(pab)-hshannon(pb)
end

function E1(θ,γ)
    # 1-body correlator
    cos(γ)*cos(2θ)
end

function E1(θ,γ,v,η)
    # noisy 1-body correlator
    η*v*E1(θ,γ)+(1-η)
end

function E2(θ,α,β)
    # 2-body correlator
    cos(α)*cos(β)+sin(α)*sin(β)*sin(2θ)
end

function E2(θ,α,β,v,η)
    # noisy 2-body correlator
    η^2*v*E2(θ,α,β)+η*(1-η)*v*E1(θ,α)+η*(1-η)*v*E1(θ,β)+(1-η)^2
end

function Pab(θ,α,β,v,η,q; binA=true, binB=false)
    # joint probabilities for Alice and Bob
    A = E1(θ,α,v,1)
    B = E1(θ,β,v,1)
    AB = E2(θ,α,β,v,1)
    # full 3x3 probability table
    P = ((η^2*(1+A+B+AB)/4, η^2*(1+(A-B)-AB)/4, η*(1-η)*(1+A)/2),
    (η^2*(1-(A-B)-AB)/4, η^2*(1-A-B+AB)/4, η*(1-η)*(1-A)/2),
    (η*(1-η)*(1+B)/2, η*(1-η)*(1-B)/2, (1-η)^2))
    # binning of Alice's outcomes
    if binA==true
        P = (P[1] .+ P[3], P[2])
    end
    # binning of Bob's outcomes
    if binB==true
        P = ((P[1][1]+P[1][3], P[1][2]),(P[2][1]+P[2][3], P[2][2]))
    end
    P
end

function hab(θ,α,β,v,η,q; binA=true, binB=false)
    # conditional entropy H(A|B)
    Pa1b3 = Pab(θ,α,β,v,η,q; binA=binA, binB=binB)
    if binA==true
        Pb3 = Pa1b3[1] .+ Pa1b3[2] 
        Pa1b3 = (Pa1b3[1]...,Pa1b3[2]...) 
    else
        Pb3 = Pa1b3[1] .+ Pa1b3[2] .+ Pa1b3[3]
        Pa1b3 = (Pa1b3[1]...,Pa1b3[2]...,Pa1b3[3]...)
    end
    hcond(Pa1b3,Pb3)
end

function hae(N,θ,v,η; infinite=false)
    # conditional entropy H(A|E)
    if η>1/(N*v) && v>=1/N && infinite==false
        ( N*(v+η)-2-sqrt(N^2*(v-η)^2+4*N*(1-v)*(1-η)) )/(2*N-2)*hbin( cos(θ)^2 )
    elseif infinite==true
        η*hbin(cos(θ)^2)
    else 
        0
    end
end

function keyrate(θ,v,η,N; binA=true, binB=false, infinite=false)
    # computes keyrate
    ae=hae(N,θ,v,η; infinite=infinite)
    ab=hab(θ,0,0,v,η,0; binA=binA, binB=binB)
    ae-ab,ae,ab
end

function best_qstrat(x0,v,η,N; binA=true, binB=false, infinite=false)
    # optimizes keyrate over partial entanglement angle
    res=Optim.optimize(x->-keyrate(x...,v,η,N; binA=binA, binB=binB, infinite=infinite)[1],x0)
    output=keyrate(Optim.minimizer(res)...,v,η,N; binA=binA, binB=binB, infinite=infinite)
    output[1],Optim.minimizer(res)
end

function draw(v,N; binA=true, binB=false, infinite=false)
    # plot key rate as function of η
    x=[1-i/1000 for i in 0:1000]
    y=zeros(1001)
    x0=[pi/4]
    for i in 1:1001
        a = best_qstrat(x0,v,x[i],N; binA=binA, binB=binB, infinite=infinite)
        y[i] = a[1]
        if a[1]<=0
            println(x[i-1])
            break
        end
        x0 = a[2]
    end
    plot(x,y)
end

function points(v,N; name="file.txt", binA=true, binB=false, infinite=false, max_ent=false)
    # saves in a file keyrate as function of η
    steps=1000
    x=[1-i/steps for i in 0:steps]
    y=zeros(steps+1)
    k=1
    x0=[pi/4]
    count=1
    while k>0
        if max_ent==true
            k=keyrate(x0...,v,x[count],N; binA=binA, binB=binB, infinite=infinite)[1]
        else
            k,x0=best_qstrat(x0,v,x[count],N; binA=binA, binB=binB, infinite=infinite)
        end
        y[count]=k
        count=count+1
    end
    open(name, "a") do io
        for i in 1:(count-2)
            println(io,x[i],"\t",y[i])
        end
    end    
    print(x[count-1],x0)
    plot(x,y)
end

function region(N; name="file.txt", binA=true, binB=false, infinite=false, max_ent=false, steps=1000, points=1000)
    # plots threshold of parameter region where the attack is allowed and writes points in a text
    eta = [1-i/steps for i in 0:steps]
    x = [1-i/points for i in 0:points] #visibility
    y = ones(points+1)
    for i in 1:(points+1)
        k=1; count=1
        x0=[pi/4]
        while k>0
            k,x0 = best_qstrat(x0,x[i],eta[count],N; binA=binA, binB=binB, infinite=infinite)
            count = count+1
        end
        y[i] = eta[count-1]
        if y[i]>=0.999999
            break
        end
    end
    open(name, "a") do io
        count=1
        while y[count]<1
            println(io,x[count],"\t",y[count])
            count=count+1
        end
    end 
    plot(x,y, xlabel="v", ylabel="η")
end