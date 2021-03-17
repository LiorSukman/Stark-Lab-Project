% UHIST         count of the unique values in a vector / cell array
%
% call          [ H, U ] = UHIST( X )
%
% returns       row vectors

% 22-apr-10 ES
%
% revisions
% 07-jan-13 extended to cell arrays

function [ h, u, id ] = uhist( x, remnans )

out = nargout > 2;
if nargin < 2
    remnans = 0;
end

h = [];
id = [];
if size( x, 2 ) == 1
    col = 1;
else
    col = 0;
end
x = x( : )';
if remnans
    nans = isnan( x );
    nnans = sum( nans );
    x( nans ) = [];
end
u = unique( x );
nu = length( u );
num = isnumeric( x );
if ~num && ~isa( x, 'cell' )
    return
end
id = zeros( size( x ) );
h = zeros( 1, nu );
for i = 1 : nu 
    if num
        idx = x == u( i );
    else
        idx = ismember( x, u{ i } );
    end
    h( i ) = sum( idx );
    if out
        id( idx ) = i;
    end
end
if col
    id = id';
end

return