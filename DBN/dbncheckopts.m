function  dbncheckopts( opts,valid_fields )
%DBNCHECKOPTS checks the validity of the opts struct
%
% see also, DBNSETUP, DBNTRAIN, DBNCREATEOPTS
%
% Copyright Søren Sønderby july 2014
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  ) 

switch func2str(opts.train_func)
    case 'rbmgenerative'
    case 'rbmdiscriminative'
        has_classes(opts)
    case 'rbmhybrid'
        has_classes(opts)
    case 'rbmsemisuplearn'
        if opts.classRBM == 0
            error('Semisupervised training without labels does not make sense, use RBMGENERATIVE')
        end
        has_classes(opts)
        assert(~isempty(opts.x_semisup));
    otherwise
        error('unknown training function')
end



    function has_classes(opts)
        % require classRBM and train labels
        assert(all([opts.classRBM == 1,...
                    ~isempty(opts.y_train) ]));
    end

end

