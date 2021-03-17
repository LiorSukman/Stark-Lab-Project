% READSPK           load spikes from a binary file
%
% GENERAL           optimized for huge files, for loading all or any subset of spikes efficiently
%
% FILE STRUCTURE    spk by spk; in each spk - sample by sample; in each sample - channel by  channel:
%
%                   s1c1t1 s1c2t1 ... s1cNt1 // spike 1, sample 1
%                   s1c1t2 s1c2t2 ... s1cNt2 // spike 1, sample 2
%                   ... 
%                   s1c1tN s1c2tN ... s1cNtN // spike 1, sample N
%                   ...
%                   sMc1t1 sMc2t1 ... sMcNt1 // spike M, sample 1
%                   ...
%                   sMc1tN sMc2tN ... sMcNtN // spike M, sample N
%
% CALL              DATA = READSPK( FNAME, NCHANS, NSAMPLES, SPIKENUMS, SOURCE, TARGET )
%
% GETS              FNAME       full file name
%                   NCHANS      {4}; number of channels / spike 
%                   NSAMPLES    {32}; number of samples / channel / spike
%                   SPIKENUMS   {[]}; vector of indices; empty loads all
%                   SOURCE      {'int16'}
%                   TARGET      {'single'}
%
% RETURNS           DATA        3D array of NCHANS x NSAMPLES x length( SPIKENUMS )
%
% CALLS             nothing
%
% SEE ALSO          READBIN
%
% NOTE              if many discontiguous spikes are needed, it may be more
%                   efficient to load all and then subsample. 

% 01-mar-11 ES

function spk = readspk( fname, nchans, nsamples, spikenums, source, target )

% arguments
nargs = nargin;
if nargs < 1 || isempty( fname ), error( 'missing arguments' ), end
if nargs < 2 || isempty( nchans ), nchans = 4; end
if nargs < 3 || isempty( nsamples ), nsamples = 32; end
if nargs < 4, spikenums = []; end
if nargs < 5 || isempty( source ), source = 'int16'; end
if nargs < 6 || isempty( target ), target = 'single'; end
if nchans <= 0 || nchans ~= round( nchans ), error( 'nchans should be a non-negative integer' ), end
if nsamples <= 0 || nsamples  ~= round( nsamples  ), error( 'nsamples should be a non-negative integer' ), end

% build the type casting string
precision = sprintf( '%s=>%s', source, target );

% determine number of bytes/sample
a = ones( 1, 1, source );
sourceinfo = whos( 'a' );
nbytes = sourceinfo.bytes;

% check input file
if ~exist( fname, 'file' )
    fprintf( 1, 'missing file %s\n', fname )
    spk = [];
    return
end
fileinfo = dir( fname );
totspikes = floor( fileinfo( 1 ).bytes / nbytes / nchans / nsamples );

% open file for reading
fp = fopen( fname, 'r' );
if fp == -1, error( 'fopen error' ), end
    
if isempty( spikenums )
    % read all spikes
    spk = fread( fp, [ nchans inf ], precision );
    nspks = size( spk, 2 ) / nsamples;
else
    % read subsets
    spikenums = spikenums( : ).';
    ridx = spikenums > totspikes;
    if sum( ridx )
        fprintf( 1, 'removing %d spikes from file %s\n', sum( ridx ), fname )
        spikenums( ridx ) = [];
    end
    [ spikenums spkidx ] = sort( spikenums );
    nspks = length( spikenums );
    grouping = cumsum( diff( [ spikenums( 1 ) spikenums ] ) > 1 ) + 1;
    groupstart = spikenums( logical( [ 1 diff( grouping ) ] ) );        % 1st spike in each group
    startposition = nbytes * nchans * nsamples * ( groupstart - 1 );    % index of 1st spike
    groupsize = hist( grouping, unique( grouping ) );                   % number of spikes in each group
    ngroups = length( groupsize );                                      % number of groups
    
    % initialize output
    spk = zeros( nchans, nsamples * sum( groupsize ) );
    eval( sprintf( 'spk = %s( spk );', target ) );
    
    % enable loading of discontiguous spikes
    n = 0;
    for gi = 1 : ngroups
        datasize = [ nchans nsamples * groupsize( gi ) ];
        rc = fseek( fp, startposition( gi ), 'bof' );
        if rc, error( 'fseek error' ), end
        tmp = fread( fp, datasize, precision );
        spk( :, n + ( 1 : nsamples * groupsize( gi ) ) ) = tmp;
        n = n + nsamples * groupsize( gi );
    end

end

% close file
fclose( fp );

% organize output
spk = reshape( spk, [ nchans nsamples nspks ] );
if ~isempty( spikenums ) && sum( diff( spkidx ) ~= 1 )
    spk = spk( :, :, spkidx );
end

return

% EOF
