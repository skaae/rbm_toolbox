function  dbncheckopts( opts,valid_fields,x )
%DBNCHECKOPTS checks the validity of the opts struct
%
% see also, DBNSETUP, DBNTRAIN, DBNCREATEOPTS
%
% Copyright Søren Sønderby july 2014
fields = fieldnames(opts);
assert(  isequal(sort(fields),sort(valid_fields))  )




valid = @(f) isfield(opts,f) == 1 && ~isempty(opts.(f));
reqboth =@(a,b) valid(a) && ~valid(b);

% check that if validation sets are given they have both x and y
if  reqboth('y_val','x_val')
    error('For validation specify both y_val and x_val')
else
    assert(size(opts.y_val,1) == size(opts.x_val,1))
end


% check if y is given if class rbm + check y size if x is given
if opts.classRBM == 1
    if ~valid('y_train')
        error('classRBM  requires y_train to be specified in opts')
    elseif exist('x','var')
        assert(size(opts.y_train,1) == size(x,1))
    end
    
end



switch func2str(opts.train_func)
    case 'rbmgenerative'
    case 'rbmdiscriminative'
        isclassRBM(opts)
        has_classes(opts)
    case 'rbmhybrid'
        isclassRBM(opts)
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

    function isclassRBM(opts)
        if opts.classRBM == 0
            error([func2str(opts.train_func) ' must be a class RBM'])
        end
    end
end

