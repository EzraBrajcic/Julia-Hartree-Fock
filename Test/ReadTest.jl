include("../src/lib/DataTypes.jl")
include("../src/lib/Reader.jl")

function main()
    BFs = ReadAtomBFs("Ti_3F.dat")
    println(typeof(BFs))
    for i in 1:lastindex(BFs)
        println(BFs[i], '\n')
    end
end

if !isdefined(Base, :test)
    main()
end