ang = 2:2:180; % orientations
centers = 4:4:180; % preferred orientations

amp = 1;%abs(abs(sind(2*centers))-1).^2; % cardinal biases

k = 2; % von-mises concentration parameter
npop = exp(k.*cosd(2*(ang - centers'))) .* (amp'+2);

sigma = 0;
npop = npop ./ (sigma+sum(npop,1)); % normalize population response

% simulate attentional gain
gain = 1;
attn = 1 + gain .* cosd(ang-45).^16;

npop_attn = npop .* attn(ismember(ang,centers))';

%% simulate triad responses
tic
all_stim = combvec(1:length(ang),1:length(ang),1:length(ang));
all_p0 = nan(length(ang),length(ang),length(ang));
all_p1 = nan(length(ang),length(ang),length(ang));
all_p2 = nan(length(ang),length(ang),length(ang));
for ii=1:length(all_stim)
    this_resp = npop_attn(:,all_stim(:,ii));
    norm_resp = 3 .* this_resp ./ sum(this_resp(:));
    all_p1(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(this_resp(:,1)-this_resp(:,2));
    all_p2(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(norm_resp(:,1)-norm_resp(:,2));

    this_resp0 = npop(:,all_stim(:,ii));
    all_p0(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(this_resp0(:,1)-this_resp0(:,2));
end
toc

c = nan(3,10);
for dd=2:10
    [mds0,c(1,dd)] = mdscale(mean(all_p0,3),dd,'Options',statset('MaxIter',1e4));
    [mds1,c(2,dd)] = mdscale(mean(all_p1,3),dd,'Options',statset('MaxIter',1e4));
    [mds2,c(3,dd)] = mdscale(mean(all_p2,3),dd,'Options',statset('MaxIter',1e4));
end

mds0 = mdscale(mean(all_p0,3),4,'Options',statset('MaxIter',1e4));
mds1 = mdscale(mean(all_p1,3),4,'Options',statset('MaxIter',1e4));
mds2 = mdscale(mean(all_p2,3),4,'Options',statset('MaxIter',1e4));

[~,mds1P] = procrustes(mds0,mds1,'Scaling',false);
[~,mds2P] = procrustes(mds0,mds2,'Scaling',false);

% 2-D plots
figure, hold on
plot(mds0(:,1),mds0(:,2),'k--','LineWidth',1)
plot(mds1P(:,1),mds1P(:,2),'r-','LineWidth',1)
plot(mds2P(:,1),mds2P(:,2),'b-','LineWidth',1)
plot(mds0(15:15:end,1),mds0(15:15:end,2),'k.','MarkerSize',10)
plot(mds0(45,1),mds0(45,2),'k.','MarkerSize',20)
plot(mds1P(15:15:end,1),mds1P(15:15:end,2),'r.','MarkerSize',10)
plot(mds1P(45,1),mds1P(45,2),'r.','MarkerSize',20)
plot(mds2P(15:15:end,1),mds2P(15:15:end,2),'b.','MarkerSize',10)
plot(mds2P(45,1),mds2P(45,2),'b.','MarkerSize',20)
axis equal
legend({'No attention','Gain only','Gain + Normalization'})

figure, hold on
plot(mds0(:,3),mds0(:,4))
plot(mds1P(:,3),mds1P(:,4))
plot(mds2P(:,3),mds2P(:,4))
axis equal

% lengths
figure, hold on
plot(ang(1:end-1),diag(mean(all_p0,3),1),'k--','LineWidth',1)
plot(ang(1:end-1),diag(mean(all_p1,3),1),'r-','LineWidth',1)
plot(ang(1:end-1),diag(mean(all_p2,3),1),'b-','LineWidth',1)
%axis([0 180 .02 .04])
legend({'No attention','Gain only','Gain + Normalization'})

%%
% get distance matrices
npop_distN = squareform(pdist(npop','cosine'));
npop_distA = squareform(pdist(npop_attn','cosine'));

mds_2N = mdscale(npop_distN,2);
mds_2A = mdscale(npop_distA,2);

[~,mds_2A] = procrustes(mds_2N,mds_2A); % align coordinates

figure, hold on
plot(mds_2N(:,1),mds_2N(:,2),'r-')
plot(mds_2N(ang==45,1),mds_2N(ang==45,2),'r.','MarkerSize',20)
plot(mds_2A(:,1),mds_2A(:,2),'b--')
plot(mds_2A(ang==45,1),mds_2A(ang==45,2),'b.','MarkerSize',20)
axis equal
