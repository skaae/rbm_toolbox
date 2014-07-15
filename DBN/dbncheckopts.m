function  dbncheckopts( opts,valid_fields )
%DBNCHECKOPTS checks the validity of the opts struct
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  ) 

if isequal(opts.train_func,@rbmhybrid) || isequal(opts.train_func,@discriminative)
    if opts.dropout_fraction > 0 || opts.dropout_in_fraction >0
        error('dropout does not work with discriminative / hybrid training at the moment')
    end
end
end

