% Spike sorting using binary pursuit.
%
% Implements the algorithm described in the following paper:
% Pillow, Shlens, Chichilnisky & Simoncelli (2013): A Model-Based Spike
% Sorting Algorithm for Removing Correlation Artifacts in Multi-Neuron
% Recordings. PLoS ONE 8, e62123.
%
% JY 2015-02-22  based on code by A. Ecker, algorithm modified from code by J. Pillow
% 
% Functions:
% BP constructor  
%       bp = BP('param1', value1, 'param2', value2, ...) constructs
            %   a Binary Pursuit object with the following optional parameters:
%       options:
%         'verbose' (true or false) - spit out text as it goes
%         'logging'  (true or false) - save output to a text file
%         'greediness' - modifies p(spike) to make the BP add more or less likely to add spikes
%
% Fit model (i.e. estimate waveform templates and spike times).
%   [X, U] = bp.fit(V, X0) fits the model to voltage trace V
%   using the initial spike times X0.
%       V  [nSamples x nElectrodes] - voltage trace
%       X0 [nSamples x nUnits]      - initial spike times (can be sparse)
%
%       X   Spike times (same format as input X0)
%       U   Array of waveform coefficients
%           E-by-K-by-M    E: number of basis functions/samples
%                          K: number of channels
%                          M: number erbose', true, 'logging', true, 'greediness', 1); 
% Example Call:
%
%   bp = BP('verbose', true, 'logging', true, 'greediness', 1); 
%   [X,U] = p.fit(V,X0)



classdef BP
    properties %#ok<*PROP>
        ne                  % number of electrode
        ncell               % number of neurons
        nsecs               % number of total seconds in the recording
        samprate            % sampling rate
        nsamps              % number of total samples
        nsampsPerFile       % num samples per saved (processed) electrode-data file
        nsecsPerW           % number of seconds' data to use for each estimate of spike waveform
        nsampsPerW          % num samples per waveform estimate
        nw                  % number time bins in spike waveform (MUST BE EVEN)
        nsampsPerBPchunk    % number samples in a chunk for binary pursuit sorting
        minISIms            % minimum ISI, in milliseconds
        minISIsamps         % minimum ISI, in samples
        verbose
        logging
        logFile
        greediness
        dirlist
        filelist
        nxct                % # time bins for temporal whitening filter
        nxcx                % # time bins to use while spatial whitening (not too big, <= 9)
    end
    
    methods
        
        function self = BP(varargin)
            % BP constructor
            %   bp = BP('param1', value1, 'param2', value2, ...) constructs
            %   a Binary Pursuit object with the following optional parameters:
            %
            %   ne                  number of electrode
            %   ncell               number of neurons
            %   nsecs               number of total seconds in the recording
            %   samprate            sampling rate
            %   nsamps              number of total samples
            %   nsampsPerFile       num samples per saved (processed) electrode-data file
            %   nsecsPerW           number of seconds' data to use for each estimate of spike waveform
            %   nsampsPerW          num samples per waveform estimate
            %   nw                  number time bins in spike waveform (MUST BE EVEN)
            %   nsampsPerBPchunk    number samples in a chunk for binary pursuit sorting
            %   minISIms            minimum ISI, in milliseconds
            %   minISIsamps         minimum ISI, in samples
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('samprate', 12000);
            p.addOptional('verbose', false);
            p.addOptional('logging', false);
            p.addOptional('nsampsPerFile', 20000);
            p.addOptional('nsecsPerW', 60);
            p.addOptional('nw', 30);
            p.addOptional('nsampsPerBPchunk', 10000)
            p.addOptional('minISIms', .5)
            p.addOptional('greediness', 1);
            p.addOptional('nxct', 16);
            p.addOptional('nxcx', 5);
            p.addOptional('dataDir', pwd);
            p.parse(varargin{:});
            
            self.samprate           = p.Results.samprate;
            self.verbose            = p.Results.verbose;
            self.logging            = p.Results.logging;
            self.nsampsPerFile      = p.Results.nsampsPerFile;
            self.nsecsPerW          = p.Results.nsecsPerW;
            self.nw                 = p.Results.nw;
            self.nsampsPerBPchunk   = p.Results.nsampsPerBPchunk;
            self.minISIms           = p.Results.minISIms;
            self.greediness         = p.Results.greediness;
            self.nxct               = p.Results.nxct;
            self.nxcx               = p.Results.nxcx;
            self.minISIsamps        = self.minISIms / 1e3 * self.samprate;
            dataDir = p.Results.dataDir;
            % setup directories for
            if ~isdir(dataDir)
                fprintf('BP: must supply data directory')
                return
            end
            
            self.dirlist.rawdat     = dataDir;
            self.dirlist.procdat    = [dataDir '/procdat/'];
            self.dirlist.W          = [self.dirlist.procdat 'Wraw/'];      % raw waveform estimates (pre-whitening)
            self.dirlist.Wwht       = [self.dirlist.procdat 'Wwht/'];      % waveform estimates after whitening
            self.dirlist.Ywht       = [self.dirlist.procdat 'Ywht/'];      % sparsified waveform estimates
            self.dirlist.tspEstim   = [self.dirlist.procdat 'tspEstim/'];  % spike train estimates
            
            %             % --------  Check that all dirlist have been created ------------
%             dirfields = fieldnames(self.dirlist);
%             for jj = 1:length(dirfields)
%                 dirname = self.dirlist.(dirfields{jj}); % dynamic field names (may break older matlab versions)
%                 if ~isdir(dirname);
%                     fprintf('SETSPIKESORTPARAMS: making directory ''%s''\n', dirname);
%                     mkdir(dirname);
%                 end
%             end
            
            % determine file name for log file
            if self.logging
                self.logFile = fullfile(self.dirlist.rawdat, [datestr(now, 'yyyymmdd_HHMMSS') '.log']);
            end
            
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % WORKING FILENAMES
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%             % NAMES FOR RAW DAT FILES (CHANGE THESE)
%             self.filelist.initspikes = [self.dirlist.rawdat, 'Xsp_init.mat']; % initial spike train estimate (sparse nsamps x ncell array)
%             self.filelist.Ydat = [self.dirlist.rawdat, 'Y.mat']; % initial spike train estimate (sparse nsamps x ncell array)
%             
%             % NAMES FOR PROCESSED DATA FILES (can leave)
%             self.filelist.Ywht = [self.dirlist.Ywht, 'Y_chunk%d.mat']; % initial spike train estimate (sparse nsamps x ncell array)
%             self.filelist.Wraw = [self.dirlist.W, 'Wraw_%d.mat']; % initial (pre-whitening) estimates of Waveform
%             self.filelist.Wwht = [self.dirlist.W, 'Wwht_%d.mat']; % estimates of Waveform w/ whitened data
%             self.filelist.Xhat = [self.dirlist.tspEstim, 'Xhat.mat'];
%             
%             % Function for loading raw electrode data (WRITE THIS FOR YOUR OWN DATA!)
%             loadrawY = @(twin)(loadRawElecDatWin(twin,filelist.Ydat));
%             
%             % Function for loading whitened electrode data (no need to rewrite)
%             loadwhitenedY = @(twin)(loadWhiteElecDatWin(twin,filelist.Ywht,sdat.nsampsPerBPchunk));
            
            
            
        end
        
        
        function [X, U] = fit(self, V, X)
            % Fit model (i.e. estimate waveform templates and spike times).
            %   [X, U] = self.fit(V, X0) fits the model to voltage trace V
            %   using the initial spike times X0.
            %
            %   [X, U] = self.fit(V, X0, iter) uses the specified number of
            %   iterations to fit the parameters (default = 3).
            %
            %   INPUTS
            %
            %   V       Continuous voltage signal
            %           T-by-K      T: number of time bins
            %                       K: number of channels
            %
            %   X0      Initial spike sorting result (sparse matrix, where
            %           X_ij=1 indicates a spike at sample i and neuron j)
            %           T-by-M      M: number of clusters
            %
            %   iter    Number of iterations to run
            %
            %
            %   OUTPUTS
            %
            %   X       Spike times (same format as input X0)
            %
            %   U       Array of waveform coefficients
            %           E-by-K-by-M    E: number of basis functions/samples
            %                          K: number of channels
            %                          M: number of neurons
            
            self.ne     = size(V,2);	  % number of electrode
            self.ncell  = size(X,2);	  % number of neurons
            self.nsamps = size(V,1);
            self.nsecs  = self.nsamps*self.samprate; % number of total seconds in the recording
            
            
            self.log(false, 'Starting to fit model\n')
            t0 = now;
            
            % 1. Estimate spike waveforms for each neuron using spike times provided
            U  = self.estimWaveforms(V,X);  % initial estimate of waveforms in non-whitened
            
            % 2. Estimate the temporal and spatial noise covariance and whiten the raw data
            Vw = self.whitenRaw(V,X,U);
            
            % 3. Re-estimate spike waveforms using whitened data
            Uw = self.estimWaveforms(Vw, X);
            
            % 4. Run binary pursuit: identify the spike times given waveforms and whitened data
            X = self.binarypursuit(Vw,X,Uw);
            
            self.log('Done fitting model [%.0fs]\n\n', (now - t0) * 24 * 60 * 60)
        end
        
        function [What,wwsigs]=estimWaveforms(self, Y, X)
            %  [What,wwsigs]=estimWaveforms(X, Y, nw)
            %
            %  Computes estimate of spike waveform for cells from spike trains and
            %  electrode data, using (correct) least-squares regression
            %
            %  Input:
            %  ------
            %   X [nsamps x ncells] - each column holds spike train of single neuron
            %   Y [nsamps x nelec] - raw electrode data
            %   nw [1 x 1] - number of time bins in the spike waveform
            %
            %  Output:
            %  -------
            %   What [nw x ne x ncells] - estimated waveforms for each cell
            %   sigs [ncells x 1] - posterior stdev of each neuron's waveform coefficients
            %
            % jw pillow 8/18/2014
            
            self.log(false, 'Estimating Waveforms\n')
            nw = self.nw;
            [nt,nc] = size(X); % number of time bins and number of cells
            ne = size(Y,2); % number of electrodes
            nw2 = nw/2;
            
            % Compute blocks for covariance matrix XX and cross-covariance XY
            XXblocks = zeros(nc*nw,nc);
            XY = zeros(nc*nw,ne);
            for jj = 1:nw
                inds = ((jj-1)*nc+1):(jj*nc);
                XXblocks(inds,:) = X(1:end-jj+1,:)'*X(jj:end,:); % spike train covariance
                XY(inds,:) = X(max(1,nw2-jj+2):min(nt,nt-jj+nw2+1),:)'*...
                    Y(max(1,jj-nw2):min(nt,nt+jj-nw2-1),:); % cross-covariance
            end
            
            % Insert blocks into covariance matrix XX
            XX = zeros(nc*nw,nc*nw);
            for jj = 1:nw
                inds1 = ((jj-1)*nc+1):(nc*nw);
                inds2 = ((jj-1)*nc+1):(jj*nc);
                XX(inds1,inds2) = XXblocks(1:(nw-jj+1)*nc,:);  % below diagonal blocks
                XX(inds2,inds1) = XXblocks(1:(nw-jj+1)*nc,:)'; % above diagonal blocks
            end
            What = XX\XY; % do regression
            What = permute(reshape(What,[nc,nw,ne]),[2,3,1]);  % reshape into tensor
            
            % 4. If desired, compute posterior variance for each waveform (function of # spikes)
            if nargout > 1
                wwsigs = sqrt(1./diag(XX(1:nc,1:nc)));
            end
            % Note: the "correct" formula should be diag(inv(Xcov)), but this is
            % close, ignoring edge effects, and much faster;
        end
        
        function [tfilts,xfilts] = computeWhiteningFilts(self, X, R)
            % Compute filters that whitens y residuals
            %
            % INPUTS:
            % ------
            %  X [nsamps x ncells] - each column holds spike train of single neuronjj
            %  R [nsamps x nelec]  - residuals
            %
            % OUTPUTS:
            % --------
            %  tfilts [nxc_t x ncells] - temporal whitening filters
            %  xfilts [nxc_x x nelec x nelec] spatial whitening filters
            self.log(false, 'Computing Whitening Filters\n')
            slen = size(X,1); % spike train data (sparse)
            ne = self.ne;
            
            % Method 1 (cost indep of nxc_t, and faster than xcorr)
            tfilts = zeros(self.nxct,ne); % temporal whitening filters
            for j = 1:ne
                % Compute autocovariance
                xc = circxcorr(R(:,j),self.nxct-1,'none');
                yxc = flipud(xc(1:self.nxct))/slen; % autocovariance
                
                % Compute whitening filt
                M = sqrtm(inv(toeplitz(yxc))); % whitening matrix
                tfilts(:,j) = M(:,self.nxct/2);
            end
            
            %  === 2. Compute spatial whitening filters ====================
            % Will be faster when we used only neighboring electrodes)
            
            % Compute temporally whiten residuals
            yresid_wht = zeros(slen,ne);  % whitened residuals
            for j = 1:ne;
                yresid_wht(:,j) = conv2(R(:,j), tfilts(:,j), 'same');
            end
            
            % Compute spatial cross-correlation(s)
            xxc = zeros(ne,ne,self.nxcx);
            xxc(:,:,1) = yresid_wht'*yresid_wht;
            for j = 2:self.nxcx
                xxc(:,:,j) = circshift(yresid_wht,j-1)'*yresid_wht;
            end
            xxc = xxc/slen;
            
%             % Compute spatial cross-correlation(s)
%             xxc2 = zeros(ne,ne,self.nxcx);
%             xxc2(:,:,1) = yresid_wht'*yresid_wht;
%             
%             sz=size(yresid_wht,1);
%             xxcI=true(sz,1);
%             xxcI2=true(sz,1);
%             for j = 2:self.nxcx
%                 xxcI(j-1)=false;
%                 xxcI2(sz-j+1)=false;
%                 xxc2(:,:,j) = yresid_wht(xxcI,:)'*yresid_wht(xxcI2,:);
%             end
% %             xxc2 = xxc2/slen;
%             grr=zeros(1,1,5);
%             grr(:)=-(-slen:-(slen-self.nxcx+1));
%             xxc2 = xxc2./repmat(grr,[4 4 1]);
%    
%             % Compute spatial cross-correlation(s)
%             xxc3 = zeros(ne,ne,self.nxcx);
%             xxc3(:,:,1) = yresid_wht'*yresid_wht;
%             
% %            for j = 1:self.nxcx 
% %             cs(:,j)=circshift((1:slen)',j-1);
% %            end
%             
%             sz=size(yresid_wht,1);
%             xxcI=true(sz,1);
% %             xxcI2=true(sz,1);
%             for j = 2:self.nxcx
%                 xxcI(j-1)=false;
%                 xxcI2(sz-j+1)=false;
%                 xxc3(:,:,j) = yresid_wht([end-j+2:end 1:end-j+1] ,:)'*yresid_wht;
%             end
%             xxc3 = xxc3/slen;
%    
            
            
            % Insert into big covariance matrix
            M = zeros(ne*self.nxcx);
            for j = 1:ne
                for i = j:ne
                    if i == j
                        jj = (j-1)*self.nxcx+1:j*self.nxcx;
                        M(jj,jj) = toeplitz(squeeze(xxc(j,j,:)));
                    else
                        jj = (j-1)*self.nxcx+1:j*self.nxcx;
                        ii = (i-1)*self.nxcx+1:i*self.nxcx;
                        M(jj,ii) = toeplitz(squeeze(xxc(j,i,:)),squeeze(xxc(i,j,:)));
                        M(ii,jj) = M(jj,ii)';
                    end
                end
            end
            % Compute filters
            Q = sqrtm(inv(M));
            xfilts = zeros(self.nxcx,ne,ne);
            for j = 1:ne
                xfilts(:,:,j) = reshape(Q(:,(j-1)*self.nxcx+ceil(self.nxcx/2)),[],ne);
            end
            
        end
        
        function Ywht = whitenRaw(self, Y,X,W)
            % [tfilts,xfilts] = compWhiteningFilts(self,X,Y,W)
            %
            %  V = self.whitenData(V, R) whitens the data V, assuming
            %   that the spatio-temporal covariance separates into a
            %   spatial and a temporal component. Whitening filters are
            %   estimated from the residuals R.
            %
            % INPUTS:
            % ------
            %  X [nsamps x ncells] - each column holds spike train of single neuronjj
            %  Y [nsamps x nelec] - raw electrode data
            %  W [nw x nelec x ncells] - tensor of spike waveforms
            %
            % OUTPUTS:
            % --------
            %  Ywht [nsamps x nelec] - whitened electrode data
            %
            %
            %  NOTES:
            % "Full" covariance too large to invert, so instead we proceed by:
            %  1. Whitening each electrode in time (using nxc_t-tap filter)
            %  2. Whiten across electrodes (using nxc_x timebins per electrode)
            %
            %  Algorithm:
            %    1. Compute the electrode residuals (raw electrodes minus spike
            %       train convolved with waveforms)
            %    2. Compute auto-correlations for each electrode
            %    3. Solve for temporal-whitening filter for each electrode
            %    4. Temporally whiten electrode residuals
            %    5. Compute spatially-whitening filters
            
            slen = size(X,1); % spike train data (sparse)
            ne = self.ne;
            
            % === 1. Compute temporal whitening filts =================
            R = self.residuals(Y,X,W);  % residuals
            
            [tfilts,xfilts] = self.computeWhiteningFilts(X, R);
            
            self.log(false, 'Whitening Raw Data...\n')
            % === 3. Now whiten the raw data ==========================
            Ywht = zeros(slen,ne);
            for ii = 1:ne  % Temporally whiten
                Ywht(:,ii) = conv2(Y(:,ii),tfilts(:,ii),'same');
            end;
            Ywht = samefilt(Ywht,xfilts,'conv');  % spatially whiten
        end
        
        function yresid  = residuals(self, Y, X, W)
            % yresid  = residuals(X, W)
            %
            % Computes sparse binary xsp convolved with ww spike waveforms ww
            %
            % INPUT:
            %   xsp [nt x nneur] - sparse binary matrix, each column is a spike train
            %   ww [nw x nelec x nneur] - tensor of spike waveforms
            %
            % OUTPUT:
            %   residuals [nt x nelec] - convolution of xsp with ww
            %
            % jw pillow 8/18/2014
            
            self.log(false, 'computing residuals...\n')
            nc = size(X,2);      % number of cells
            wwid = size(W,1)/2;    % 1/2 length of spike waveform
            iirel = (0:wwid*2-1)'; % relative time indices
            slen = size(X,1);    % number of time samples
            
            Vpred = zeros(slen+wwid*2,size(W,2));  % allocate memory (with padding at beginning and end)
            
            for j = 1:nc  % loop over neurons
                isp = find(X(:,j));  % find the spike times
                for i = 1:length(isp);
                    ii = isp(i)+iirel;
                    Vpred(ii,:) = Vpred(ii,:)+W(:,:,j);
                end
            end
            Vpred = Vpred(wwid+1:end-wwid,:);  % remove padding at the end
            
            yresid = Y - Vpred;
        end
        
        function [wproj,wnorm] = waveformConvolution(self, W)
            % [wproj,wnorm] = waveformConvolution(W)
            %
            % Computes the full time-convolution of each waveform with every other waveform (
            % (This convolution is 'valid' in the rows, and 'full' in the columns)
            %
            % INPUT:
            %   W [ntime x nelectrodes x ncells ] - tensor of spike waveforms
            %
            % OUTPUT:
            %   wproj [ 2*ntime-1 x ncells x ncells] - tensory of waveforms convolved w each other
            %   wnorm [ ncells x 1 ] - squared L2 norm of each waveform
            %
            % jw pillow 8/18/2014
            
            nw = self.nw;
            nc = size(W,3);
            
            wproj = zeros(2*nw-1,nc,nc);
            for jcell = 1:nc
                for icell = jcell:nc  % compute conv of w(icell) convolved with w(jcell)
                    zz = W(:,:,jcell)*W(:,:,icell)';
                    for j = 1:nw
                        wproj(j:j+nw-1,icell,jcell) = wproj(j:j+nw-1,icell,jcell) + zz(:,nw-j+1);
                    end
                    % for all cell pairs (not including auto-correlation)
                    if icell > jcell
                        wproj(:,jcell,icell) = flipud(wproj(:,icell,jcell));
                    end
                end
            end
            wnorm = diag(squeeze(wproj(nw,:,:)));  % dot prod of each waveform with itself
            
        end
        
        function [X] = binarypursuit(self,Y,X,W)
            % [X,nrefviols] = runBinaryPursuit(X,Y,W)
            %
            % Binary programming solution for
            %    argmin_xx  ||W*X- Y||^2
            % via greedy binary updates to X.
            %
            % Inputs:  X = initial guess at spike trains
            %          Y = raw voltages
            %          WW = waveform tensor
            %          pspike = prior probability of a spike in a bin for each cell
            %          wproj = projection of w onto itself
            %          wnorm = vector norm of each waveform
            %          minISI = minimum ISI (in number of samples)
            %
            % Outputs:
            %   X - estimated binary spike train
            %   nrefviols [1 x ncells] - number of spikes removed for violating refractory period
            %
            % This is the workhorse function that does the actual BP (generally called from
            % estimSps_BinaryPursuit.m, which handles passing in and out chunks of data).
            %
            % Also includes (kludgey) final step of removing spikes that violate minimum ISI.
            %
            % jw pillow 8/18/2014
            % jly/jk modified for speed increase
            
            [wproj,wnorm] = self.waveformConvolution(W);
            pspike = mean(X) * self.greediness;
            
            maxSteps = 2e5;  % Max # of passes (i.e., # spikes added/removed)
            
            % initialize params
            [nw,~,nc] = size(W);  % size of waveform (time bins) and number of cells
            slen = size(X,1);  % number of time bins
            nw2 = nw/2;  % half the time-length of the waveform
            
            spkthresh = -log(pspike)+log(1-pspike);  % diff in pior log-li for adding sp
            
%             % Compute residual errors between electrode data and prediction from initial spike train
%             rr = self.residuals(Y,X,W);  % residuals
%             
%             % -----------------------------------------------------------------
%             % 1. Compute MSE reduction for each spike bin position
%             
%             fprintf('estimSps_binary_chunk: initializing dlogli matrix...\n');
%             rwcnv = validfilt_sprse(rr,W);
%             
%             dlogli = rwcnv - repmat((.5*wnorm'+spkthresh),slen-nw+1,1);
%             for jcell = 1:nc
%                 ispk = find(X(nw2+1:slen-nw2+1,jcell));
%                 dlogli(ispk,jcell) = -dlogli(ispk,jcell)-wnorm(jcell);
%             end
%             dlogli = [-100*ones(nw2,nc); dlogli];
            
            %alternative
            dlogli = validfilt_sprse(Y,W);
            [ispk,icell] = find(X);
            iirel = (-nw+1:nw-1);  % indices whose dlogli affected by inserting a spike
            szd = size(dlogli);
            for iSpike=1:length(ispk)
                iichg = ispk(iSpike)+iirel-nw2;
                valid=iichg>1 & iichg<=szd(1);
                dlogli(iichg(valid),:)=dlogli(iichg(valid),:)-wproj(valid,:,icell(iSpike));
            end
            dlogli = dlogli - repmat((.5*wnorm'+spkthresh),slen-nw+1,1);

            for jcell = 1:nc
                ispk = find(X(nw2+1:slen-nw2+1,jcell));
                dlogli(ispk,jcell) = -dlogli(ispk,jcell)-wnorm(jcell);
            end
            dlogli = [-100*ones(nw2,nc); dlogli];

            self.log(true, 'Beginning greedy insertion of spikes\n')
            
            % -----------------------------------------------------------------
            % 2. Now begin maximizing posterior by greedily inserting/removing spikes
            
            % fprintf('estimSps_binary_chunk: beginning to add/subtract spikes\n');
            % modreport = 100;
            nstp = 1; % Counts # of passes through data
            iirel = (-nw+1:nw-1);  % indices whose dlogli affected by inserting a spike
            iirefract = -self.minISIsamps:self.minISIsamps;
                      
            % jake kluge for searching a subset of samples 
            szd = size(dlogli);
            possibleChanges = sparse(dlogli(:) > 0);
            
            
            [mxvals,iimx] = max(dlogli);  % Find maximum of dlogli in each column
            [mx,cellsp] = max(mxvals);    % Max of the maxima (get cell #)
            ispk = iimx(cellsp);  % The time bin to flip (add/remove spike)
                       
            while (mx > 0) && (nstp <= maxSteps);
                
                %     % Uncomment to print reports on progress
                %     % --------------------------------------
                %     if mod(nstp,modreport) == 0     % Report output only every 100 bins
                %         if (X(ispk,cellsp) == 1)
                %             fprintf('step %d: REMOVED sp, cell %d, bin %d, Dlogli=%.3f\n',...
                %                 nstp,cellsp,ispk,dlogli(ispk,cellsp));
                %         else
                %             fprintf(1, 'step %d: inserted cell %d, bin %d, Dlogli=%.3f\n',...
                %                 nstp,cellsp,ispk,dlogli(ispk,cellsp));
                %         end
                %     end % ----------------------------------
                
                % ----------------------------------------
                % 2A.  Insert or remove spike in location where logli is most improved
                if X(ispk,cellsp) == 0   % Insert spike --------------
                    
                    X(ispk,cellsp) = 1;
                    dloglictrbin = -dlogli(ispk,cellsp); % dLogli for this bin
%                     inds = ispk-nw2:ispk+nw2-1;
%                     rr(inds,:) = rr(inds,:)- W(:,:,cellsp); % remove waveform
                    
                    % update dlogli for all bins within +/- nw
                    if ((ispk+iirel(1)) >= nw2+1) && ((ispk+iirel(end)) <= slen-nw2+1)
                        iichg = ispk+iirel;
                        prevdlogli=dlogli(iichg,:);
                        dlogli(iichg,:)=  dlogli(iichg,:)-wproj(:,:,cellsp);
                    else % update only for relevant range of indices
                        iichg = ispk+iirel;
                        ii = find((iichg>=(nw2+1)) & (iichg<=(slen-nw2+1)));
                        iichg = iichg(ii);
                        prevdlogli=dlogli(iichg,:);
                        dlogli(iichg,:)= dlogli(iichg,:)-wproj(ii,:,cellsp); 
                    end
%                     dlogli(ispk,cellsp) = dloglictrbin; % set for center bin
                    dlogli(ispk+iirefract,cellsp) = min(dlogli(ispk+iirefract,cellsp),dloglictrbin); % set for center bin +iirefract
%                     
%                     for icell=1:self.ncell
%                         indsToMoveOut=dlogli(ispk,cellsp)<0
%                         
%                         
%                         indsToMoveOut=dlogli(ispk,cellsp)>0
%                     end
                    
                else   % Remove spike ----------------------------------
                    
                    X(ispk,cellsp) = 0;
                    dloglictrbin = -dlogli(ispk,cellsp); % dLogli for this bin
%                     inds = ispk-nw2:ispk+nw2-1;
%                     rr(inds,:) = rr(inds,:) + W(:,:,cellsp); % add waveform back to residuals
                    
                    % update dlogli for all bins within +/- nw
                    if ((ispk+iirel(1)) >= nw2+1) && ((ispk+iirel(end))<=slen-nw2+1)
                        iichg = ispk+iirel;
                        prevdlogli=dlogli(iichg,:);
                        dlogli(iichg,:)= dlogli(iichg,:)+wproj(:,:,cellsp);
                    else % update only for relevant range of indices
                        iichg = ispk+iirel;
                        ii = find((iichg>=(nw2+1)) & (iichg<=(slen-nw2+1)));
                        iichg = iichg(ii);
                        prevdlogli=dlogli(iichg,:);
                        dlogli(iichg,:)=  dlogli(iichg,:)+wproj(ii,:,cellsp);
                    end
%                     dlogli(ispk,cellsp) = dloglictrbin; % set for center bin
                    dlogli(ispk+iirefract,cellsp) = min(dlogli(ispk+iirefract,cellsp),dloglictrbin); % set for center bin +iirefract %also changing iirefract is dangerous as those might have lower logli and we don't want to increase the change of it crossing  0 later
                end
                               

                % ----------------------------------------
                % 2B. Do some index arithmetic to max maximum dlogli for each cell
                %  (Big speedup from searching for the max over all bins for each cell).
                
                % update possibleChanges
                [newI, newJ]=find(xor(prevdlogli>0,dlogli(iichg,:)>0));
                if ~isempty(newI)
                    newI=iichg(newI)+(newJ'-1)*szd(1);
                    possibleChanges(newI)=dlogli(newI)>0; %#ok<SPRIX>
                end

                % get new maximum
                [ii]=find(possibleChanges);
                [mx, iims] = max(dlogli(possibleChanges));
                ispk= rem(ii(iims)-1, szd(1))+1;
                cellsp = (ii(iims) - ispk)/szd(1) + 1; 
                
                if isempty(mx)
                    mx=0;
                end
                
                nstp = nstp+1;
            end
            self.log(true, 'Done\n')
            
            display(['Done in ' num2str(nstp-1) ' Iterations\n'])
            
            % Notify if MaxSteps exceeded
            if nstp > maxSteps
                fprintf('estimSps_binary_chun: max # passes exceeded (dlogli=%.3f)\n', dlogli(ispk,cellsp));
            end
            
%             % ----------------------------------------------------------
%             % 3. Finally, remove spikes that violate refractory period
%             nrefviols = zeros(1,nc);  % initialize counter
%             for jcell = 1:nc
%                 isis = diff(find(X(:,jcell)));
%                 while any(isis<minISI)
%                     tsp = find(X(:,jcell)); % spike indices
%                     nsp = length(tsp); % number of spikes
%                     badii = find(isis<minISI);
%                     badii = union(badii,badii(badii<nsp)+1); % indices of problem spikes
%                     badtsp = tsp(badii);
%                     [~,ii] = max(dlogli(badtsp,jcell)); % Find which spike is least likely
%                     ispk = badtsp(ii);
%                     
%                     % Remove this spike and update
%                     nrefviols(jcell) = nrefviols(jcell)+1; % count the ISI removal
%                     X(ispk,jcell) = 0; % remove from spike train
%                     dloglictrbin = -dlogli(ispk,jcell); % dLogli for this bin
%                     inds = ispk-nw2:ispk+nw2-1;
%                     rr(inds,:) = rr(inds,:)+ W(:,:,jcell); % add waveform back to residuals
%                     
%                     % update dlogli for all bins within +/- nw
%                     if ((ispk+iirel(1)) >= nw2+1) && ((ispk+iirel(end))<=slen-nw2+1)
%                         iichg = ispk+iirel;
%                         dlogli(iichg,:)= dlogli(iichg,:)+wproj(:,:,jcell);
%                     else % update only for relevant range of indices
%                         iichg = ispk+iirel;
%                         ii = find((iichg>=(nw2+1)) & (iichg<=(slen-nw2+1)));
%                         iichg = iichg(ii);
%                         dlogli(iichg,:)=  dlogli(iichg,:)+wproj(ii,:,jcell);
%                     end
%                     dlogli(ispk,jcell) = dloglictrbin; % set for center bin
%                     
%                     % Recompute ISIs
%                     isis = diff(find(X(:,jcell)));
%                     
%                 end
%             end
            
            
        end

        
        
        function [U, X, priors, order] = orderTemplates(self, U, X, priors, orderBy)
            % Order waveform templates spatially.
            %   [U, X, priors] = orderTemplates(self, U, X, priors, 'y')
            %   orders the waveform templates spatially by th y-location of
            %   the channel with maximum energy.
            
            M = numel(priors);
            order = self.layout.channelOrder(orderBy);
            mag = zeros(1, M);
            peak = zeros(1, M);
            for m = 1 : M
                Ui = mean(U(:, order, m, :), 4);
                [mag(m), peak(m)] = max(sum(Ui .* Ui, 1));
            end
            [~, order] = sort(peak * 1e6 - mag);
            U = U(:, :, order, :);
            X = X(:, order);
            priors = priors(order);
        end
        
        
        function y = interp(self, x, k, shape)
            % Interpolate x using subsample shifts
            %   y = self.interp(x, k) interpolates x, shifting it by k
            %   subsamples (i.e. k / self.subsampling samples).
            
            if nargin < 4
                shape = 'same';
            end
            p = self.upsamplingFactor;
            h = self.upsamplingFilter(:, ceil(p / 2) + k);
            y = convn(x, h, shape);
        end
        
    end
    
    
    
    methods (Access = private)
        
        function log(self, varargin)
            
            % first input numeric: 0 = starting / 1 = done with step
            if islogical(varargin{1})
                if ~varargin{1}
                    varargin(1) = [];
                    tic
                else
                    varargin{1} = 'done [%.1fs]\n';
                    varargin{2} = toc;
                end
            end
            
            % write to log file?
            if self.logging
                fid = fopen(self.logFile, 'a');
                assert(fid > 0, 'Failed to open log file %s!', self.logFile)
                fprintf(fid, varargin{:});
                fclose(fid);
            end
            
            % print to command line?
            if self.verbose
                fprintf(varargin{:})
            end
        end
        
    end
end
