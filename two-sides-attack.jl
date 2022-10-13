# This program assumes that Eve attacks either Alice or Bob

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

function part_ent(θ)
    # creates density matrix of a partially entangled state
    z=[1 0]
    u=[0 1]
    ψ=cos(θ)*kron(z,z)+sin(θ)*kron(u,u)
    ψ'*ψ
end

function hab(θ,α,β,η,Q)
    #computes H(A|B) in case we use postselection
    ρ=part_ent(θ)
    X = [ 0 1 ; 1 0 ]
    Z = [ 1 0 ; 0 -1 ]
    A=cos(α)*Z+sin(α)*X
    B=cos(β)*Z+sin(β)*X
    POVM=[[η*0.5*(Matrix(I,2,2)+A), η*0.5*(Matrix(I,2,2)-A), (1-η)*Matrix(I,2,2)],[η*0.5*(Matrix(I,2,2)+B), η*0.5*(Matrix(I,2,2)-B), (1-η)*Matrix(I,2,2)]]
    P=[[ tr(kron(POVM[1][j],POVM[2][i])*ρ) for i=1:3] for j=1:3]
    ω=[[1,Q,Q],[Q,Q^2,Q^2],[Q,Q^2,Q^2]]
    pν=sum(sum(ω[i][j]*P[i][j] for i=1:3) for j=1:3)
    Ph=[[P[1][1]/pν P[1][2]*Q/pν P[1][3]*Q/pν],[(P[2][1]+P[3][1])*Q/pν (P[2][2]+P[3][2])*Q^2/pν (P[2][3]+P[3][3])*Q^2/pν]]
    xlog2x(Ph[1][1])+xlog2x(Ph[1][2])+xlog2x(Ph[1][3])+xlog2x(Ph[2][1])+xlog2x(Ph[2][2])+xlog2x(Ph[2][3])-xlog2x(Ph[1][1]+Ph[2][1])-xlog2x(Ph[1][2]+Ph[2][2])-xlog2x(Ph[1][3]+Ph[2][3])
end

function hae(N,θ,v,η; infinite=false)
    # H(A|E) without postselection
    p = N*η*v*(1-η)/(N*v-1)
    q = (1-η)^2
    if infinite==true 
        η^2*hbin( cos(θ)^2 )
    elseif infinite==false && η>=2/(N+1)
        (1-2*p-q)*hbin( cos(θ)^2 )
    else
        0
    end
end

function hae(N,θ,v,η,Q; infinite=false)
    # H(A|E) in case we use postselection
    p = N*η*(1-η)/(N-1)
    q = (1-η)^2
    if infinite==true 
        η^2*hbin( Q*cos(θ)^2/(1-(1-Q)*cos(θ)^2) )
    elseif infinite==false && η>=2/(N+1)
        (1-2*p-q)*hbin( Q*cos(θ)^2/(1-(1-Q)*cos(θ)^2) )
    else
        0
    end
end

function keyrate(θ,v,η,N; binA=true, binB=false, infinite=false)
    # computes keyrate without postselection
    ae=hae(N,θ,v,η; infinite=infinite)
    ab=hab(θ,0,0,v,η,0; binA=binA, binB=binB)
    ae-ab,ae,ab
end

function keyrate(θ,Q,v,η,N; binA=true, binB=false, infinite=false)
    # computes keyrate with postselection
    ae=hae(N,θ,v,η,cos(Q)^2; infinite=infinite)
    ab=hab(θ,0,0,η,cos(Q)^2)
    ae-ab,ae,ab
end

function best_qstrat(x0,v,η,N; binA=true, binB=false, infinite=false)
    # optimizes keyrate over θ if x0 is 1-dim and also over postselection if x0 is 2-dim
    res=Optim.optimize(x->-keyrate(x...,v,η,N; binA=binA, binB=binB, infinite=infinite)[1],x0)
    output=keyrate(Optim.minimizer(res)...,v,η,N; binA=binA, binB=binB, infinite=infinite)
    output[1],Optim.minimizer(res)
end

function draw(v,N; binA=true, binB=false, infinite=false, max_ent=false, postselection=false)
    # plot key rate as a function of η
    x=[1-i/1000 for i in 0:1000]
    y=zeros(1001)
    x0=[pi/4]
    if postselection==true
        x0=[pi/8,pi/3]
    end
    for i in 1:1001
        if max_ent==true
            a = keyrate(pi/4,v,x[i],N; binA=binA, binB=binB, infinite=infinite)
        else
            a = best_qstrat(x0,v,x[i],N; binA=binA, binB=binB, infinite=infinite)
        end
        y[i] = a[1]
        if a[1]<=0
            println(x[i-1],a[2])
            break
        end
        if x[i]<=0.4
            x0 = a[2]
        end
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
            k = keyrate(x0...,v,x[count],N; binA=binA, binB=binB, infinite=infinite)[1]
        else
            k,x0 = best_qstrat(x0,v,x[count],N; binA=binA, binB=binB, infinite=infinite)
        end
        y[count]=k
        count=count+1
    end
    open(name,"a") do io
        for i in 1:(count-2)
            println(io,x[i],"\t",y[i])
        end
    end    
    println(x[count-1],x0)
    plot(x,y)
end

function region(N; name="file.txt", binA=true, binB=false, infinite=false, max_ent=false, enhanced=false, steps=1000, points=1000)
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
