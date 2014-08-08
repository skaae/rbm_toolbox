function [ drbm ] = cpRBMtoGPU( hrbm)
fld = fields(hrbm);
drbm = struct();
for i=1:numel(fld)
    fieldName = fld{i};
    if isnumeric(hrbm.(fieldName))
        drbm.(fieldName) = gpuArray(hrbm.(fieldName) );
    else
        drbm.(fieldName) = hrbm.(fieldName);
    end
end