library(NMF)
library(ggplot2)

dirpath = "D:\\workspace"
setwd(dirpath)

datapath = "./gradients.csv"


gradient_mci = read.csv(datapath, header=TRUE)
gradient_mci = gradient_mci[, 6:ncol(gradient_mci)]
dim(gradient_mci)

normalize = function(x) {
  return ((x-min(x)) / (max(x) - min(x)))
}

gradient_mci = apply(gradient_mci, 2, normalize)
gradient_mci_t = t(gradient_mci)

estim.r = nmf(gradient_mci_t, 2:6, method="nsNMF",nrun=30, seed=123456)
saveRDS(estim.r, "./estim_r.RDATA")

coph <- estim.r$measures$cophenetic
coph.diff=coph[1:length(coph)-1]-coph[2:length(coph)]
k.best=which.max(coph.diff)+1
print(k.best)
plot(2:6,coph,type="b",col="purple")

rss <- estim.r$measures$rss
rss.diff=rss[1:length(rss)-1]-rss[2:length(rss)]
estim.best.r = nmf(gradient_mci_t, k.best, method="nsNMF",nrun=30, seed=123456)
saveRDS(estim.best.r, "./estim_r_best.RDATA")


W = basis(estim.best.r)
H = coef(estim.best.r)
write.csv(W, "./NMF_W.csv", row.names = FALSE, col.names = FALSE)
write.csv(H, "./NMF_H.csv", row.names = FALSE, col.names = FALSE)

heatmap(W,Rowv = NA,Colv = NA,revC = TRUE)