using BEAST, CompScienceMeshes, LinearAlgebra, Test

@testset "EFIE Float32 far assembly retains signed triangle precision" begin
    mesh64 = meshicosphere(12,1.0)
    X64 = raviartthomas(mesh64)
    pairs = reduce(hcat,([fn[1].cellid,fn[2].cellid] for fn in X64.fns))
    X32 = raviartthomas(convert(Float32,mesh64),pairs)
    rows = findall(p -> p[1]>0.9,X64.pos)
    cols = findall(p -> p[1]<-0.9,X64.pos)
    @test min(length(rows),length(cols)) > 50
    qs = BEAST.DoubleNumQStrat(2,3)
    A64 = zeros(ComplexF64,length(rows),length(cols))
    A32 = zeros(ComplexF32,size(A64))
    op64 = Maxwell3D.singlelayer(;wavenumber=2π/10)
    op32 = Maxwell3D.singlelayer(;wavenumber=Float32(2π/10))
    asm64 = BEAST.blockassembler(op64,X64,X64;quadstrat=qs)
    asm32 = BEAST.blockassembler(op32,X32,X32;quadstrat=qs)
    asm64(rows,cols,(v,m,n)->(A64[m,n]+=v))
    asm32(rows,cols,(v,m,n)->(A32[m,n]+=v))
    @test eltype(A32) == ComplexF32
    typed = zeros(ComplexF32,size(A32))
    typedstore(v::ComplexF32,m,n) = (typed[m,n]+=v)
    asm32(rows,cols,typedstore)
    @test typed == A32
    @test norm(ComplexF64.(A32)-A64)/norm(A64) < 1e-5
    # The same physical mesh/basis in Float32 should not create a singular-value tail.
    function frobeniusrank(A,tol)
        s = svdvals(ComplexF64.(A))
        return findfirst(r -> norm(s[r+1:end])<=tol*norm(s),0:length(s))-1
    end
    @test abs(frobeniusrank(A32,1e-4)-frobeniusrank(A64,1e-4)) <= 1
    @test BEAST.scalartype(op32,X32,X32) == ComplexF32
    @test eltype(first(asm32.scratchpool.available).zlocal) == ComplexF64
    @test BEAST._assemblytype(op32,ComplexF32,BEAST.DoubleNumSauterQstrat(2,3,5,5,4,3)) == ComplexF32
    @test BEAST._assemblytype(Maxwell3D.doublelayer(;wavenumber=Float32(2π/10)),ComplexF32,qs) == ComplexF32
    testcells = BEAST.active_element_ids(X32,rows)
    trialcells = BEAST.active_element_ids(X32,cols)
    tad = BEAST.reduce_assembly_data(asm32.testassemblydata,rows,testcells)
    bad = BEAST.reduce_assembly_data(asm32.trialassemblydata,cols,trialcells)
    for i in unique([1,cld(length(rows),2),length(rows)])
        row = zeros(ComplexF32,1,length(cols))
        BEAST.assemblerow_body!(op32,BEAST.subset(X32,rows[i:i]),asm32.testelements,refspace(X32),
            bad,X32,view(asm32.trialelements,trialcells),refspace(X32),zeros(ComplexF32,3,3),
            (asm32.quadraturedata[1],view(asm32.quadraturedata[2],:,trialcells)),
            (v,m,n)->(row[m,n]+=v);quadstrat=qs)
        @test row ≈ A32[i:i,:] rtol=5eps(Float32)
    end
    for j in unique([1,cld(length(cols),2),length(cols)])
        col = zeros(ComplexF32,length(rows),1)
        BEAST.assemblecol_body!(op32,tad,X32,view(asm32.testelements,testcells),refspace(X32),
            BEAST.subset(X32,cols[j:j]),asm32.trialelements,refspace(X32),zeros(ComplexF32,3,3),
            (view(asm32.quadraturedata[1],:,testcells),asm32.quadraturedata[2]),
            (v,m,n)->(col[m,n]+=v);quadstrat=qs)
        @test col ≈ A32[:,j:j] rtol=5eps(Float32)
    end
    # Dense assembly must also finish the signed sum before its Float32 store.
    dense = assemble(op32,BEAST.subset(X32,rows),BEAST.subset(X32,cols);quadstrat=qs)
    @test eltype(dense) == ComplexF32
    @test dense ≈ A32 rtol=5eps(Float32)
    colored = assemble(op32,BEAST.subset(X32,rows),BEAST.subset(X32,cols);
        threading=:cellcoloring,quadstrat=qs,scheduler=BEAST.OhMyThreads.DynamicScheduler())
    @test colored ≈ A32 rtol=5eps(Float32)
    # Preserve additive storage and safe reuse after a failing store.
    twice = copy(A32)
    asm32(rows,cols,(v,m,n)->(twice[m,n]+=v))
    @test twice ≈ 2A32 rtol=5eps(Float32)
    @test_throws ErrorException asm32(rows,cols,(args...)->error("store failed"))
    repeated = [Threads.@spawn begin
        A = zeros(ComplexF32,size(A32))
        asm32(rows,cols,(v,m,n)->(A[m,n]+=v))
        A
    end for _ in 1:4]
    @test all(A -> isapprox(A,A32;rtol=5eps(Float32)),fetch.(repeated))
end
