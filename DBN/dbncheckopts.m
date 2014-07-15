function  dbncheckopts( opts,valid_fields )
%DBNCHECKOPTS checks the validity of the opts struct
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  ) 

end

