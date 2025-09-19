%% neural population models of representational geometry

% define the feature space
ang = -89:90; % orientations
centers = -86:4:90; % preferred orientations

k = 4; % von-mises concentration parameter
npop = exp(k.*cosd(2*(ang - centers')));

sigma = 0;
npop = npop ./ (sigma+sum(npop,1)); % normalize population response

% simulate attentional gain
gain = 0.6;
n = 20; % control width of attention
targ = -45; % target orientation
attn = 1 + gain .* cosd(ang-targ).^n;

npop_attn = npop .* attn(ismember(ang,centers))';

%% simulate triad responses

triad_stim = -89:90; % orientations to simulate triads over
all_stim = combvec(1:length(triad_stim),1:length(triad_stim),1:length(triad_stim));

% preallocate arrays
all_p0 = zeros(length(triad_stim).*ones(1,3));
all_p1 = zeros(length(triad_stim).*ones(1,3));
all_p2 = zeros(length(triad_stim).*ones(1,3));

for ii=1:length(all_stim)
    % get population response to the triad
    [this_stim,~] = find(triad_stim(all_stim(:,ii))==ang');
    this_resp = npop_attn(:,this_stim);
    norm_resp = 3 .* this_resp ./ sum(this_resp,'all');

    all_p1(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(this_resp(:,1)-this_resp(:,2));
    all_p2(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(norm_resp(:,1)-norm_resp(:,2));

    % response in the population without attention
    this_resp0 = npop(:,this_stim);
    all_p0(all_stim(1,ii),all_stim(2,ii),all_stim(3,ii)) = vecnorm(this_resp0(:,1)-this_resp0(:,2));
end

% multidimensional scaling
mds0 = mdscale(mean(all_p0,3,'omitnan'),4,'Options',statset('MaxIter',1e4));
mds1 = mdscale(mean(all_p1,3,'omitnan'),4,'Options',statset('MaxIter',1e4));
mds2 = mdscale(mean(all_p2,3,'omitnan'),4,'Options',statset('MaxIter',1e4));

[~,mds1P] = procrustes(mds0,mds1,'Scaling',false);
[~,mds2P] = procrustes(mds0,mds2,'Scaling',false);

% collate solutions
d = nan([size(mds0) 3]);
d(:,:,1) = mds0;
d(:,:,2) = mds1P;
d(:,:,3) = mds2P;

% 2-D plots
figure, hold on
p1 = plot(squeeze([d(:,1,:);d(1,1,:)]),squeeze([d(:,2,:);d(1,2,:)]),'LineWidth',1);
p2 = plot(squeeze(d(15:15:end,1,:)),squeeze(d(15:15:end,2,:)),'.','MarkerSize',10);
p3 = plot(squeeze(d(45,1,:)),squeeze(d(45,2,:)),'.','MarkerSize',20);
p1(1).Color = 'k';
p1(1).LineStyle = '--';
p1(2).Color = 'r';
p1(3).Color = 'b';
axis([-.3 .3 -.3 .3])
axis equal
legend({'No attention','Gain only','Gain + Normalization'})

% sliding window
clear cond_length
wind = 20; % ±4 stim around each orientation
for ii=length(triad_stim):-1:1
    this_set = mod(ii-wind:wind+ii,length(triad_stim));
    this_set(this_set==0) = length(triad_stim);
    cond_length(:,ii) = squeeze(sum(vecnorm(diff(d(this_set,:,:),[],1),2,2),1));
end

figure, hold on
p = plot(triad_stim,cond_length,'LineWidth',1);
p(1).Color = 'k';
p(1).LineStyle = '--';
p(2).Color = 'r';
p(3).Color = 'b';
axis([-90 90 .3 .6])
legend({'No attention','Gain only','Gain + Normalization'})
