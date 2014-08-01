function [ hrbm ] = cpRBMtoHost( drbm )
fld = fields(drbm);
drbm = struct();
for i=1:numel(fld)
    fieldName = fld{i};
    drbm.(fieldName) = gather(hrbm.(fieldName) );
end
drbm.gpu = 0;