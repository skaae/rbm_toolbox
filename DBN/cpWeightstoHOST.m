function [ hrbm ] = cpWeightstoHOST( drbm )
% copy weights to host for testing
hrbm = struct();
hrbm.W = drbm.gather(drbm.W );
hrbm.U = drbm.gather(drbm.U );
hrbm.b = drbm.gather(drbm.b );
hrbm.c = drbm.gather(drbm.c );
hrbm.d = drbm.gather(drbm.d );

hrbm.errfunc = drbm.gather(drbm.errfunc );
hrbm.reconerror = drbm.gather(drbm.reconerror );
hrbm.valerror = drbm.gather(drbm.valerror );
hrbm.trainerror = drbm.gather(drbm.trainerror );


