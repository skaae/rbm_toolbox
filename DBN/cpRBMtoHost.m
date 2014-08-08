function [ hrbm ] = cpRBMtoHOST( drbm )
fld = fields(drbm);
hrbm = struct();
for i=1:numel(fld)
    fieldName = fld{i};
    hrbm.(fieldName) = gather(drbm.(fieldName) );
end