function  dbncheckopts( opts,valid_fields )
%DBNCHECKOPTS checks the validity of the opts struct
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  ) 

switch func2str(opts.train_func)
    case 'rbmgenerative'
    case 'rbmdiscriminative'
        has_classes(opts)
    case 'rbmhybrid'
        has_classes(opts)
    case 'rbmsemisuplearn' 
        has_classes(opts)
        assert(~isempty(opts.x_semisup));
        if isequal(opts.semisup_type,@rbmgenerative) && opts.classRBM == 0
            error('semisupervised training without labels does not make sense, use rbmgenerative')
        end
    otherwise
        error('unknown training function')
end



    function has_classes(opts)
        % require classRBM and train labels
        assert(all([opts.classRBM == 1,...
                    ~isempty(opts.y_train) ]));
    end

end

