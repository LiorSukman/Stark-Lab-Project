% KRUSKALWALLIS2        2-way non-parametric ANOVA with interaction term
%
% call                  [ P, INFO ] = KRUSKALWALLIS2( Y, G1, G2, RANKF )
%
% gets                  Y           values
%                       G1          group 1 classification ('rows')
%                       G2          group 2 classification ('columns')
%                       RANKF       flag {1}
%
% returns               P           row, col, interaction (column vector)
%                       INFO        fields: 1-way ANOVA:    tab1, F1, p1
%                                           2-way ANOVA:    tab2, F2, p2
%                                           2-way KW:       chi_stat, p
%
% calls                 RANKCOLS
%
% example
%
%       y = [ 709 679 699 657 594 677 592 538 476 508 505 539 ];
%       y = reshape( y, 6, 2 );
%       s = [ ones( 3, 2 ); 2 * ones( 3, 2 ) ];
%       f = ones( 6, 1 ) * [ 1 2 ];
%       kruskalwallis2( y, s, f )
%
% reference: Sokal & Rohlf 2001 p.324 (2way), p. 424-426 (KW), p.445-447 (KW2)

% 11-aug-04 ES

% revisions
% 18-aug-04 tie correction

function [ p, info ]            = kruskalwallis2( y, g1, g2, rankf )

if nargin < 3
    error( 'input size mismatch' )
end
if nargin < 4 || isempty( rankf )
    rankf                       = 1;
end

% initialize
DV                              = NaN;
dummy                           = DV * ones( 3, 1 );
p                               = dummy;
info                            = struct( 'tab1', DV * ones( 3 ), 'F1', DV, 'p1', DV...
    , 'tab2', DV * ones( 5, 3 ), 'F2', dummy, 'p2', dummy...
    , 'chi_stat', dummy, 'p', dummy );
if rankf
    y                           = rankcols( y( : ) );
else
    y                           = y( : );
end

if any( isnan( y ) ) || all( y == y( 1 ) )
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% go over subgroups
ug1                             = unique( g1( : )' );
ug2                             = unique( g2( : )' );
r                               = length( ug1 );                            % 'rows'
c                               = length( ug2 );                         	% 'columns'
rc                              = r * c;                                    % n = unique( ngroup ); 
N                               = numel( y );                               % for unbalanced; balanced: N = rc * n; 
gmeans                          = NaN( length( ug1 ), length( ug2 ) );
ss_within                       = gmeans;
ngroup                          = gmeans;
for i                           = ug1                                       % 'rows'
    for j                       = ug2                                       % 'columns'
        idx                     = g1 == i & g2 == j;
            gmeans( i, j )      = nanmean( y( idx ) );                    	% Y |
            ss_within( i, j )   = nansum( ( y( idx ) - gmeans( i, j ) ) .^ 2 );  % error
            ngroup( i, j )      = nansum( nansum( idx ) );
    end
end
cmeans                          = nanmean( gmeans, 1 );                     % C |
rmeans                          = nanmean( gmeans, 2 );                     % R |
cnums                           = nansum( ngroup, 1 );
rnums                           = nansum( ngroup, 2 );
m                               = nanmean( y( : ) );                        % Y ||
SS_subgr                        = nansum( nansum( ngroup .* ( gmeans - m ) .^ 2 ) ); % extension to unbalanced
SS_within                       = nansum( nansum( ss_within ) );            % error

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1way (as if prod( nu ) different groups); SS is:
%   total = subgroup + within (error)
tab1                            = [ rc - 1 SS_subgr; N - rc SS_within ];
tab1( :, 3 )                    = tab1( :, 2 ) ./ tab1( :, 1 );
tab1                            = [ tab1; nansum( tab1 ) ];
tab1( end )                     = NaN;
F1                              = tab1( 1, 3 ) / tab1( end - 1, 3 );
p1                              = 1 - fcdf( F1, tab1( 1, 1), tab1( end - 1, 1) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2way (rows, cols, inter); SS is partitioned as follows:
%   subgroup = row + col + inter
%   total = subgroup + error (within)
SS_rows                         = nansum( rnums .* ( rmeans - m ) .^ 2 );
SS_cols                         = nansum( cnums .* ( cmeans - m ) .^ 2 );
SS_inter                        = SS_subgr - SS_rows - SS_cols;
tab2( :, 1 )                    = [ r - 1; c - 1; ( r - 1 ) * ( c - 1 ); N - rc ];
tab2( :, 2 )                    = [ SS_rows; SS_cols; SS_inter; SS_within ];
tab2( :, 3 )                    = tab2( :, 2 ) ./ tab2( :, 1 );
tab2                            = [ tab2; nansum( tab2 ) ];
tab2( end )                     = NaN;
F2                              = tab2( 1 : 3, 3 ) / tab2( end - 1, 3 );
p2                              = 1 - fcdf( F2, tab2( 1 : 3, 1), tab2( end - 1, 1) );                % row (g1), column (g2), interaction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% non parametric
if rankf                                                                    % rank specific stuff
    MS_tot                      = tab2( 5, 2 ) / tab2( 5, 1 );                               % ~N * ( N + 1 ) / 12, with tie correction
    chi_stat                    = tab2( 1 : 3, 2 ) / MS_tot;
    p                           = 1 - chi2cdf( chi_stat, tab2( 1 : 3, 1 ) );
    info.chi_stat               = chi_stat;
    info.p                      = p;
else
    p                           = p2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% summarize
info.tab1                       = tab1;
info.F1                         = F1;
info.p1                         = p1;
info.tab2                       = tab2;
info.F2                         = F2;
info.p2                         = p2;

return

% EOF
