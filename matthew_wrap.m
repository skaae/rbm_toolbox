function [cleave_error, other_measures ] = matthew_wrap( confusion )
[~, other_measures] = matthew(confusion);

cleave_error = 1-other_measures(2);

end

