function nll = pred_triad_angdist(beh,k)

n = length(beh);

if size(k,1)==1
    k = repmat(k,[3 1]);
end

% predict triad responses for each subject
ns = nan(1,n);
all_nll = nan(1,n);

for ii=1:n
    ns(ii) = length(beh(ii).triadRefA);
    beh(ii).sid = ii.*ones(1,ns(ii));
end

all_triadRef = [beh.triadRefA];
all_triadChoice = vertcat(beh.triadChoiceA);
all_triadChosenLoc = [beh.triadChosenLoc];
all_sid = [beh.sid];

y = all_triadChosenLoc==1; % response vector

angdist = @(a,b) min(mod(a-b,180),mod(b-a,180)); % angular distance function
distMat0 = angdist(0:5:175,(0:5:175)').*pi/90 .* shiftdim(k',-2); % arc length
distMat = [distMat0(:,:,:,1)  zeros(36,36,n)    zeros(36,36,n);
            zeros(36,36,n)   distMat0(:,:,:,2)  zeros(36,36,n);
            zeros(36,36,n)    zeros(36,36,n)   distMat0(:,:,:,3)];

iL = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,1)',all_sid);
iR = sub2ind(size(distMat),all_triadRef,all_triadChoice(:,2)',all_sid);

% get choice probabilities based on distances of current coordinates
p = normcdf(sqrt(2).*(distMat(iR)-distMat(iL)));
p(p<1e-16) = 1e-16; p(p>1-1e-16) = 1-1e-16;

for ii=n:-1:1
    all_nll(ii) = -(log2(p(all_sid==ii))*(y(all_sid==ii))' + log2(1-p(all_sid==ii))*(1-y(all_sid==ii))');
end

nll = sum(all_nll./ns).*mean(ns);
