function [ drbm ] = cpRBMtoGPU( hrbm)
fld = fields(hrbm);
drbm = struct();
for i=1:numel(fld)
    fieldName = fld{i};
    drbm.(fieldName) = gpuArray(hrbm.(fieldName) );;
end
drbm.gpu = gpuArray(1);