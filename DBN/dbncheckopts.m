function  dbncheckopts( opts,valid_fields )
%DBNCHECKOPTS checks the validity of the opts struct
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  ) 

if isequal(opts.train_func,@rbmsemisuplearn)
    assert(~isempty(opts.x_semisup))  % required for semisup learning
end
end

