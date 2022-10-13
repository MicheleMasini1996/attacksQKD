# QKD attacks
Julia code to compute upper bounds on keyrate for specific protocols

To use the code, it is necessary to have installed the Julia packages LinearAlgebra, Optim and Plots. To do this, launch Julia and type

    using Pkg
    Pkg.add(["LinearAlgebra", "Optim", "Plots"])

It is now possible to use the code. Import one of the files (depending on whether you wish to allow attacks on one or both sides) using

    include("one-side-attack.jl")

or

    include("two-sides-attack.jl")

You can now get a plot of your keyrate with visibility 1 and 2 measurements typing 

    draw(1,2)

You can change the settings of the bin (on Alice or Bob's side), plot the case of a number of measurements tending to infinite, or fix the initial state to be a maximally entangled state by calling the function in the following way

    draw(1,2; binA=true, binB=false, infinite=false, max_ent=false)

and changing the trues or falses. For the two-sides-attack it is possible to compute the keyrate also with postselection by typing

    draw(1,2; postselection=true)

Finally, we can plot the threshold of the region of parameters η and v where our attacks are allowed by typing

    region(2; binA=true, binB=false, infinite=false, max_ent=false, postselection=false) # replace 2 with the number of measurements you want
